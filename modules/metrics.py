import numpy as np
import pandas as pd
from collections import Counter
from .common import Phase, Reliability, MetricResult, SwingEvent

def calculate_angle(p1_x, p1_y, p2_x, p2_y):
    """Calculates the angle between two points in degrees relative to the horizon."""
    if pd.isna([p1_x, p1_y, p2_x, p2_y]).any():
        return np.nan
    return np.degrees(np.arctan2(p2_y - p1_y, p2_x - p1_x))

def calculate_shoulder_angle(row):
    """Calculates the angle of the shoulder line."""
    return calculate_angle(row.get('L_SHOULDER_x'), row.get('L_SHOULDER_y'),
                           row.get('R_SHOULDER_x'), row.get('R_SHOULDER_y'))

def calculate_hip_angle(row):
    """Calculates the angle of the hip line."""
    return calculate_angle(row.get('L_HIP_x'), row.get('L_HIP_y'),
                           row.get('R_HIP_x'), row.get('R_HIP_y'))

def calculate_swing_metrics(df, swing: SwingEvent):
    """
    Evaluates biomechanical metrics for a single SwingEvent.
    Returns a list of MetricResult objects.
    """
    results = []
    
    # Extract data for the key frames of the swing
    addr_row = df.iloc[swing.indices['addr']]
    top_row = df.iloc[swing.indices['top']]
    imp_row = df.iloc[swing.indices['imp']]
    
    # ---------------------------------------------------------
    # 1. TEMPO RATIO
    # ---------------------------------------------------------
    tempo_flag = "Good" if 2.5 <= swing.tempo <= 3.5 else "Needs Attention"
    results.append(MetricResult(
        name="Tempo Ratio",
        value=round(swing.tempo, 1),
        unit=":1",
        phase=Phase.TIMING,
        reliability=Reliability.HIGH if swing.tempo > 0 else Reliability.INVALID,
        flag=tempo_flag
    ))
    
    # ---------------------------------------------------------
    # 2. SHOULDER TURN (Top of backswing vs Address)
    # ---------------------------------------------------------
    top_shoulder = calculate_shoulder_angle(top_row)
    addr_shoulder = calculate_shoulder_angle(addr_row)
    
    if not np.isnan(top_shoulder) and not np.isnan(addr_shoulder):
        # Calculate delta, adjusting for 360-degree wrap-around
        turn = abs((top_shoulder - addr_shoulder + 180) % 360 - 180)
        results.append(MetricResult(
            name="Shoulder Turn",
            value=round(turn, 1),
            unit="deg",
            phase=Phase.TOP,
            reliability=Reliability.HIGH,
            flag="Good" if turn > 80 else "Restricted"
        ))
        
    # ---------------------------------------------------------
    # 3. HIP SWAY (Lateral movement from Address to Top)
    # ---------------------------------------------------------
    addr_hip_x = (addr_row.get('L_HIP_x', 0) + addr_row.get('R_HIP_x', 0)) / 2
    top_hip_x = (top_row.get('L_HIP_x', 0) + top_row.get('R_HIP_x', 0)) / 2
    
    if not pd.isna([addr_hip_x, top_hip_x]).any():
        sway = top_hip_x - addr_hip_x
        results.append(MetricResult(
            name="Hip Sway",
            value=round(sway, 1),
            unit="px",
            phase=Phase.TOP,
            reliability=Reliability.MED,
            flag="Excessive" if abs(sway) > 30 else "Stable"
        ))
        
    # ---------------------------------------------------------
    # 4. X-FACTOR PROXY (Shoulder vs Hip separation at Top)
    # ---------------------------------------------------------
    top_hip = calculate_hip_angle(top_row)
    
    if not np.isnan(top_shoulder) and not np.isnan(top_hip):
        x_factor = abs((top_shoulder - top_hip + 180) % 360 - 180)
        results.append(MetricResult(
            name="X-Factor (Proxy)",
            value=round(x_factor, 1),
            unit="deg",
            phase=Phase.TOP,
            reliability=Reliability.MED,
            flag="Strong Coil" if x_factor > 40 else "Weak Coil"
        ))
        
    return results

def print_summary_statistics(all_results):
    """
    Parses a list of MetricResult objects across multiple swings 
    and prints a formatted statistical summary.
    """
    if not all_results:
        print("No metrics available to summarize.")
        return

    print("\n" + "="*70)
    print("📊 BIOMECHANICAL SWING SUMMARY")
    print("="*70)
    
    # Group metrics by name
    metric_groups = {}
    flags = []
    
    for res in all_results:
        if res.reliability != Reliability.INVALID:
            if res.name not in metric_groups:
                metric_groups[res.name] = []
            metric_groups[res.name].append(res.value)
            if res.flag and res.flag not in ["Good", "Stable", "Strong Coil"]:
                flags.append(f"{res.name}: {res.flag}")

    # Print averages and ranges
    for name, values in metric_groups.items():
        if values:
            avg = np.mean(values)
            std = np.std(values) if len(values) > 1 else 0
            min_v = np.min(values)
            max_v = np.max(values)
            
            # Find the unit by looking up the first result with this name
            unit = next(r.unit for r in all_results if r.name == name)
            
            print(f"{name:<22} : {avg:>6.1f} {unit:<4} (±{std:.1f})  [Range: {min_v:.1f} to {max_v:.1f}]")
            
    # Print coaching flags
    if flags:
        print("\n⚠️  Most Common Coaching Flags:")
        flag_counts = Counter(flags)
        for flag, count in flag_counts.most_common(3):
            print(f"  • {flag} ({count}x)")
            
    print("="*70 + "\n")
