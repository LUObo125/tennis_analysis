from ultralytics import YOLO 

model = YOLO('yolov8x')

result = model.track('input_videos/input_video.avi',conf=0.2, save=True, device=0)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)