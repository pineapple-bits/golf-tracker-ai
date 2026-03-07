from ultralytics import YOLO
from modules.vision import process_video, CameraValidator
from modules.biomech import weighted_smooth, detect_swings
from modules.common import MP_INDICES

# 1. Load
model = YOLO("model/path")

# 2. See
raw_data = process_video("video/path", model, MP_INDICES, CameraValidator())

# 3. Think
clean_data = weighted_smooth(raw_data, MP_INDICES)
swings = detect_swings(clean_data, CameraValidator().roll_angle)

# 4. Result
print(f"Analysis Complete: Found {len(swings)} swings.")
