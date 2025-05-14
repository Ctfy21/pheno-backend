from ultralytics import YOLO

model = YOLO("D:/python_projects/pheno-backend/services/train/seed/weights/best.pt")

def find_seeds(path_to_images: str) -> list[int]:
    results = model(path_to_images, conf=0.37, save=True)
    counting_result = []
    for result in results:
        boxes = len(result.boxes.xywh)
        counting_result.append(boxes)
    return counting_result

