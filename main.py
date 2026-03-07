import argparse
import matplotlib.pyplot as plt
from ultralytics import YOLO

from modules.common import MP_INDICES
from modules.vision import CameraValidator, process_video
from modules.physics import weighted_smooth, detect_swings
from modules.metrics import calculate_swing_metrics, print_summary_statistics

def main():
    parser = argparse.ArgumentParser(description="AI Golf Swing Tracker & Biomechanical Analyzer")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (e.g., data/raw/swing.mp4)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to the YOLO club detection model weights")
    args = parser.parse_args()

    print("\n⛳ Initializing AI Golf Tracker...")
    

    print(f"Loading YOLO model from: {args.model}")
    club_model = YOLO(args.model)
    cam_validator = CameraValidator()


    print(f"\n🎬 Processing Video: {args.video}")
    df_raw = process_video(args.video, club_model, MP_INDICES, cam_validator)
    
    if df_raw is None or len(df_raw) == 0:
        print("❌ Failed to extract data from video. Exiting.")
        return


    print("\n🧹 Applying Data Smoothing...")
    df = weighted_smooth(df_raw, MP_INDICES)

    plt.figure(figsize=(15, 5))
    df['avg_wrist_y'] = (df['L_WRIST_y'] + df['R_WRIST_y']) / 2
    plt.plot(df['frame'], df['avg_wrist_y'], linewidth=2)
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Wrist Y Position (pixels)', fontsize=12)
    plt.title('Wrist Height Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('wrist_movement.png', dpi=150)
    print("✅ Saved wrist movement visualization to 'wrist_movement.png'")

    print("\n🔍 Detecting Swings...")
    swings = detect_swings(df, cam_validator.roll_angle, prominence=40, min_height=50)

    if not swings:
        print("❌ No valid swings detected in the video.")
        return


    print("\n📈 Calculating Biomechanical Metrics...")
    all_results = []
    for swing in swings:
        results = calculate_swing_metrics(df, swing)
        all_results.extend(results)

    print_summary_statistics(all_results)


if __name__ == "__main__":
    main()
