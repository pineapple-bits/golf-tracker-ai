import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from .common import SwingEvent

def weighted_smooth(df, mp_indices):
    """Apply smoothing to landmark data."""
    df = df.copy()
    
    # 1. NaN out low confidence
    for name in mp_indices:
        mask = df[f'{name}_vis'] < 0.4
        df.loc[mask, [f'{name}_x', f'{name}_y', f'{name}_z']] = np.nan
    
    # 2. Interpolate
    df = df.interpolate(limit=4, limit_direction='both')
    
    # 3. SavGol filter
    for col in [c for c in df.columns if '_x' in c or '_y' in c or '_z' in c]:
        if len(df) > 15 and df[col].notna().sum() > 15:
            try:
                df[col] = savgol_filter(df[col].fillna(method='ffill').fillna(method='bfill'), 7, 2)
            except:
                pass
    
    print("✅ Smoothing applied!")
    return df

def detect_swings(df, cam_roll_angle, prominence=40, min_height=50):
    """Detect swing events - looks for HIGHEST wrist points (top of backswing)."""
    swings = []
    df['avg_wrist_y'] = (df['L_WRIST_y'] + df['R_WRIST_y']) / 2
    if len(df) < 30:
        print("❌ Not enough frames for swing detection")
        return []
    
    fps = df.at[0, 'fps']
    wy = df['avg_wrist_y'].copy()
    
    # Fill NaN values
    wy = wy.ffill().bfill()
    
    # Calculate velocity (rate of change of wrist Y position)
    velocity = np.gradient(wy)
    
    # Smooth velocity to reduce noise
    if len(velocity) > 11:
        velocity = savgol_filter(velocity, 11, 3)
    
    # Calculate 10-second cooldown in frames
    cooldown_frames = int(fps * 10)
    
    # Find peaks - INVERTED because we want HIGHEST points (smallest Y values)
    # In image coordinates, smaller Y = higher position
    peaks, properties = find_peaks(-wy, distance=fps*0.5, prominence=prominence, height=-wy.max())
    
    print(f"\n🔍 Swing Detection:")
    print(f"   Found {len(peaks)} potential swing peaks (top of backswing)")
    print(f"   Peak frames: {peaks}")
    if len(peaks) > 0:
        print(f"   Peak heights (wrist Y): {[f'{wy[p]:.1f}' for p in peaks]}")
        print(f"   Peak prominences: {properties['prominences']}")
    
    # Track the last swing's end frame for cooldown
    last_swing_end = -cooldown_frames  # Allow first swing immediately
    
    for peak_idx, top in enumerate(peaks):
        print(f"\n   Analyzing peak {peak_idx+1} at frame {top}...")
        
        # Check cooldown period
        if top < last_swing_end + cooldown_frames:
            frames_until_ready = (last_swing_end + cooldown_frames) - top
            time_until_ready = frames_until_ready / fps
            print(f"      ⏸️ COOLDOWN: Skipping peak (need {time_until_ready:.1f}s more)")
            continue
        
        # Find address using VELOCITY - look for near-zero velocity before top
        # Address is where the golfer is stationary/setup before starting backswing
        addr = 0
        search_back = min(top, int(fps * 10.0))  # Look back up to 2 seconds
        velocity_threshold = 0.5  # Pixels per frame - adjust based on your video
        
        # Start from a bit before top and go backwards
        for i in range(top - int(fps * 0.2), max(0, top - search_back), -1):
            # Look for point where velocity is near zero (stationary setup)
            if abs(velocity[i]) < velocity_threshold:
                addr = i
                print(f"      Found low velocity at frame {i} (vel: {velocity[i]:.2f})")
                break
        
        # If no low-velocity point found, use a default lookback
        if addr == 0:
            addr = max(0, top - int(fps * 1.0))
            print(f"      No clear address found, using default {fps*1.0:.1f}s before top")
        
        print(f"      Address: frame {addr} (wrist Y: {wy[addr]:.1f}, velocity: {velocity[addr]:.2f})")
        print(f"      Top: frame {top} (wrist Y: {wy[top]:.1f}, velocity: {velocity[top]:.2f})")
        
        # Find impact - go forward from top to find LOWEST wrist point
        # Impact should have highest Y value (lowest position)
        search_forward = min(len(df) - top, int(fps * 0.5))
        
        if search_forward < 5:
            print(f"      ⚠️ Skipping - not enough frames after top")
            continue
        
        # Find the maximum Y value (lowest position) after top
        impact_window = wy[top:top+search_forward]
        imp_local = impact_window.idxmax()
        imp = imp_local
        
        print(f"      Impact (lowest wrist): frame {imp} (wrist Y: {wy[imp]:.1f}, velocity: {velocity[imp]:.2f})")
        
        # Refine with club detection if available
        club_search_start = max(0, imp-3)
        club_search_end = min(len(df), imp+3)
        club_win = df['CLUB_y'].iloc[club_search_start:club_search_end]
        
        if club_win.notna().any():
            club_imp = club_win.idxmax()  # Highest Y = lowest position for club too
            if abs(club_imp - imp) < 5:  # Only use if close to our estimate
                print(f"      Club-refined impact: frame {club_imp}")
                imp = club_imp
        
        # Calculate tempo
        back_t = (top - addr) / fps
        down_t = (imp - top) / fps
        tempo = back_t / down_t if down_t > 0.01 else 0
        
        print(f"      Backswing: {back_t:.2f}s, Downswing: {down_t:.2f}s")
        print(f"      Tempo: {tempo:.1f}:1")
        
        # Determine swing type based on total movement
        swing_height = abs(wy[imp] - wy[top])  # Distance from top to impact
        type_s = "Full" if swing_height > min_height else "Partial"
        
        print(f"      Swing amplitude: {swing_height:.1f}px")
        print(f"      Type: {type_s}")
        
        # Validate the swing makes sense
        if back_t < 0.3:  # Backswing too short (at least 0.3 seconds)
            print(f"      ⚠️ Skipping - backswing too short ({back_t:.2f}s)")
            continue
        
        if imp <= top:  # Impact should be after top
            print(f"      ⚠️ Skipping - impact before top")
            continue
        
        # Additional validation - tempo should be reasonable
        if tempo < 1.5 or tempo > 5.0:
            print(f"      ⚠️ Skipping - unrealistic tempo ({tempo:.1f}:1)")
            continue
        
        # Validate downswing duration
        if down_t < 0.1 or down_t > 0.6:
            print(f"      ⚠️ Skipping - unrealistic downswing duration ({down_t:.2f}s)")
            continue
        
        swing = SwingEvent(
            id=len(swings)+1,
            indices={'addr': addr, 'top': top, 'imp': imp},
            type=type_s,
            tempo=tempo,
            cam_roll=cam_roll_angle
        )
        swings.append(swing)
        print(f"      ✅ Swing #{len(swings)} added!")
        
        # Update cooldown - start from impact + 1 second follow-through buffer
        follow_through_buffer = int(fps * 1.0)  # 1 second buffer
        last_swing_end = imp + follow_through_buffer
        cooldown_end_time = (last_swing_end + cooldown_frames - imp) / fps
        print(f"      🕐 Cooldown activated: {cooldown_end_time:.1f}s until next detection allowed")
    
    print(f"\n✅ Total swings detected: {len(swings)}")
    return swings