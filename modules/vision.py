import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from ultralytics import YOLO


class CameraValidator:
    def __init__(self):
        self.feature_params = dict(maxCorners=50, qualityLevel=0.3, minDistance=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, 
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None
        self.p0 = None
        self.shake_detected = False
        self.roll_angle = 0.0

    def check_roll(self, frame):
        """Estimates horizon roll angle using Hough Lines."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        
        angles = []
        if lines is not None:
            for rho, theta in lines[:, 0]:
                deg = np.degrees(theta)
                diff_v = abs(deg - 0)
                diff_h = abs(deg - 90)
                diff_v2 = abs(deg - 180)
                
                if diff_v < 15: angles.append(0 - deg if deg < 90 else 180 - deg)
                elif diff_v2 < 15: angles.append(180 - deg)
                elif diff_h < 15: angles.append(90 - deg)

        if angles:
            self.roll_angle = np.mean(angles)
        else:
            self.roll_angle = 0.0
        return self.roll_angle

    def check_stability(self, frame, mask_rect=None):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.p0 is None:
            self.prev_gray = frame_gray
            mask = np.ones_like(frame_gray, dtype=np.uint8) * 255
            if mask_rect: cv2.rectangle(mask, mask_rect[0], mask_rect[1], 0, -1)
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **self.feature_params)
            return True

        if self.p0 is None or len(self.p0) < 5: return True
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.p0, None, **self.lk_params)
        
        if p1 is not None:
            good_new = p1[st==1]
            good_old = self.p0[st==1]
            if len(good_new) > 0:
                drift = np.linalg.norm(good_new - good_old, axis=1)
                if np.mean(drift) > 2.5: self.shake_detected = True
                self.prev_gray = frame_gray
                self.p0 = good_new.reshape(-1, 1, 2)
        return not self.shake_detected


def process_video(video_path, club_model, mp_indices, cam_validator):
    """Extract landmarks and club positions from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video!")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video Info: {w}x{h} @ {fps:.2f} FPS")
    
    # Init Camera
    ret, f0 = cap.read()
    if ret:
        cam_validator.check_roll(f0)
        mask_rect = ((int(w*0.3), int(h*0.1)), (int(w*0.7), int(h*0.9)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    data = []
    f_idx = 0
    
    # Setup MediaPipe
    mp_pose_solution = mp.solutions.pose
    
    with mp_pose_solution.Pose(min_detection_confidence=0.5, model_complexity=2, 
                                smooth_landmarks=True) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            if f_idx % 10 == 0: 
                cam_validator.check_stability(frame, mask_rect)
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)
            row = {'frame': f_idx, 'fps': fps}
            
            # 1. Landmarks
            if res.pose_landmarks:
                for name, id in mp_indices.items():
                    lm = res.pose_landmarks.landmark[id]
                    row[f'{name}_x'] = lm.x * w
                    row[f'{name}_y'] = lm.y * h
                    row[f'{name}_z'] = lm.z * w
                    row[f'{name}_vis'] = lm.visibility
            else:
                for name in mp_indices:
                    row[f'{name}_x'] = np.nan
                    row[f'{name}_vis'] = 0

            # 2. Club Detection
            yolo = club_model(frame, verbose=False)
            cx, cy, cconf = np.nan, np.nan, 0.0
            
            if len(yolo[0].boxes) > 0:
                box = yolo[0].boxes.xywh.cpu().numpy()[0]
                cx, cy = int(box[0]), int(box[1])
                cconf = float(yolo[0].boxes.conf.cpu().numpy()[0])
                
                # Sanity check
                if not np.isnan(row.get('R_SHOULDER_x', np.nan)) and not np.isnan(row.get('R_WRIST_x', np.nan)):
                    sx, sy = row['R_SHOULDER_x'], row['R_SHOULDER_y']
                    wx, wy = row['R_WRIST_x'], row['R_WRIST_y']
                    arm_len = np.linalg.norm([wx-sx, wy-sy])
                    club_dist = np.linalg.norm([cx-sx, cy-sy])
                    if club_dist > (arm_len * 2.5):
                        cconf = 0.0
            
            row['CLUB_x'] = cx
            row['CLUB_y'] = cy
            row['CLUB_conf'] = cconf
            data.append(row)
            f_idx += 1
            
            if f_idx % 50 == 0:
                print(f"   Processed {f_idx} frames...")

    cap.release()
    df = pd.DataFrame(data)
    print(f"✅ Video processing complete! {len(df)} frames extracted")
    return df