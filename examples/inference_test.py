from ultralytics import YOLO

model = YOLO("models/best.pt") 
results = model.predict('input_vids/input.mp4', save=True, project = 'RAPID-Project/runs')

print(results[0])
print("Inference complete. Results saved.")

for box in results[0].boxes:
    print(f"Class: {box.cls}, Confidence: {box.conf}, BBox: {box.xyxy}")

    