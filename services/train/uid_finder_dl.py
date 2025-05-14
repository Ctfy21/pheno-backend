from ultralytics import YOLO
import numpy as np
model = YOLO("D:/python_projects/pheno-backend/services/train/digits/weights/best.pt")

def number_to_uid(numbers, boxes_x_min, boxes_x_max):
    if len(numbers) < 6:
        return "0"
    elif len(numbers) > 8:
        return "0"
    elif len(numbers) == 6:
        return f"{''.join([str(box) for box in numbers[0:4]])}.{''.join([str(box) for box in numbers[4:5]])}.{''.join([str(box) for box in numbers[5:6]])}"
    elif len(numbers) == 7:
        difference = []
        for i in range(len(boxes_x_min)-1):
            difference.append((boxes_x_min[i+1] - boxes_x_max[i]))
        diff_indexed = np.argsort(difference[4:6])
        return f"{''.join([str(box) for box in numbers[0:4]])}.{''.join([str(box) for box in numbers[4:5 + diff_indexed[len(diff_indexed) - 1]]])}.{''.join([str(box) for box in numbers[5 + diff_indexed[len(diff_indexed) - 1]:7]])}"
    elif len(numbers) == 8:
        return f"{''.join([str(box) for box in numbers[0:4]])}.{''.join([str(box) for box in numbers[4:6]])}.{''.join([str(box) for box in numbers[6:8]])}"


def find_uid(path_to_images: str) -> list[str]:
    results = model(path_to_images, conf=0.5, save=True, show_conf=False)
    uid_list = []
    for result in results:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        arr_boxes = boxes.numpy()
        row_boxes_x_min = arr_boxes[:, 0]
        row_boxes_x_max = arr_boxes[:, 2]
        class_boxes = classes.numpy()
        # Sort indices based on row_boxes values
        sorted_indices = np.argsort(row_boxes_x_min)
        sorted_class_boxes = class_boxes[sorted_indices]
        sorted_boxes_x_min = row_boxes_x_min[sorted_indices]
        sorted_boxes_x_max = row_boxes_x_max[sorted_indices]
        uid = number_to_uid(sorted_class_boxes.astype(int), sorted_boxes_x_min, sorted_boxes_x_max)
        uid_list.append(uid)
    return uid_list






