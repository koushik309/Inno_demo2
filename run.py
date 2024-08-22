import cv2
import os
import argparse
from pathlib import Path
import shutil
from passing import detect  # Ensure this is imported correctly
from concurrent.futures import ProcessPoolExecutor, as_completed

def split_image(img_path, stride=32):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"The image at {img_path} could not be loaded. Please check the path and file.")
    h, w, _ = img.shape
    pad_h = (stride - h % stride) % stride
    pad_w = (stride - w % stride) % stride
    padded_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    h, w = padded_img.shape[:2]
    return [padded_img[:h//2, :w//2], padded_img[:h//2, w//2:], padded_img[h//2:, :w//2], padded_img[h//2:, w//2:]]

def save_image_parts(img_parts, temp_dir):
    part_paths = []
    for idx, part in enumerate(img_parts):
        part_path = os.path.join(temp_dir, f"part_{idx + 1}.jpg")
        cv2.imwrite(part_path, part)
        part_paths.append(part_path)
    return part_paths

def parse_predictions(pred_path):
    predictions = []
    with open(pred_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            if len(data) == 6:
                predictions.append([int(data[0])] + list(map(float, data[1:])))
            elif len(data) == 5:
                predictions.append([int(data[0])] + list(map(float, data[1:])) + [1.0])
    return predictions

def draw_predictions_on_original(image, predictions, part_idx, img_shape, part_shape):
    h_half, w_half = img_shape[0] // 2, img_shape[1] // 2
    x_offset = (part_idx % 2) * w_half
    y_offset = (part_idx // 2) * h_half

    bboxes = []
    for pred in predictions:
        class_id, x_center, y_center, width, height, conf = pred
        abs_x_center = x_center * part_shape[1] + x_offset
        abs_y_center = y_center * part_shape[0] + y_offset
        abs_width = width * part_shape[1]
        abs_height = height * part_shape[0]

        top_left = (int(abs_x_center - abs_width / 2), int(abs_y_center - abs_height / 2))
        bottom_right = (int(abs_x_center + abs_width / 2), int(abs_y_center + abs_height / 2))

        if class_id == 0:
            color = (0, 255, 0)  # Green for Healthy
            label = "Healthy"
        else:
            color = (0, 0, 255)  # Red for Sick
            label = "Sick"

        cv2.rectangle(image, top_left, bottom_right, color, 2)
        cv2.putText(image, f"{label} {conf:.2f}", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        norm_x_center = abs_x_center / img_shape[1]
        norm_y_center = abs_y_center / img_shape[0]
        norm_width = abs_width / img_shape[1]
        norm_height = abs_height / img_shape[0]

        bboxes.append((class_id, norm_x_center, norm_y_center, norm_width, norm_height, conf))

    return bboxes

def process_part(part_path, idx, full_img, img_shape, part_shape, args):
    # Create the Namespace object with all necessary attributes
    opt = argparse.Namespace(
        weights=args.weights,
        source=part_path,
        img_size=args.img_size,
        save_txt=True,
        save_conf=True,
        project=args.temp_dir,  # Use temp_dir for the project during detection
        name='outputs',
        exist_ok=True,
        conf_thres=0.25,
        iou_thres=0.45,
        device='',
        view_img=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        update=False,
        no_trace=True
    )
    
    # Perform the detection and handle the predictions
    detect(opt)
    pred_path = os.path.join(args.temp_dir, 'outputs', 'labels', os.path.basename(part_path).replace('.jpg', '.txt'))
    predictions = parse_predictions(pred_path) if os.path.exists(pred_path) else []
    return draw_predictions_on_original(full_img, predictions, idx, img_shape, part_shape)

def process_image_parts(part_paths, full_img_path, args):
    full_img = cv2.imread(full_img_path)
    img_shape = full_img.shape
    part_shape = (img_shape[0] // 2, img_shape[1] // 2)

    all_bboxes = []

    with ProcessPoolExecutor() as executor:
        futures = []
        for idx, part_path in enumerate(part_paths):
            # Pass the args to process_part
            futures.append(executor.submit(process_part, part_path, idx, full_img, img_shape, part_shape, args))

        for future in as_completed(futures):
            bboxes = future.result()
            all_bboxes.extend(bboxes)

    labels_dir = os.path.join(args.project, args.name, 'labels')
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

    bbox_txt_path = os.path.join(labels_dir, os.path.basename(full_img_path).replace('.jpg', '.txt'))
    with open(bbox_txt_path, 'w') as f:
        for bbox in all_bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')
    print(f"Bounding box values saved to: {bbox_txt_path}")

def clear_labels_directory(temp_dir):
    labels_dir = os.path.join(temp_dir, 'outputs', 'labels')
    if os.path.exists(labels_dir):
        shutil.rmtree(labels_dir)
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Source image path or directory')
    parser.add_argument('--weights', nargs='+', type=str, default=['yolov7.pt'], help='Path(s) to model weights')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size (pixels)')
    parser.add_argument('--save-txt', action='store_true', help='Save results to .txt file')
    parser.add_argument('--save-conf', action='store_true', help='Save confidence scores')
    parser.add_argument('--project', type=str, default='results', help='Project directory')
    parser.add_argument('--name', type=str, default='run', help='Run name')
    parser.add_argument('--temp-dir', type=str, default='temp_parts', help='Temporary directory for image parts')
    parser.add_argument('--no-trace', action='store_true', help='Do not trace the model')  # Add this line

    args = parser.parse_args()

    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.project, args.name)).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(args.source):
        for filename in os.listdir(args.source):
            file_path = os.path.join(args.source, filename)
            if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Processing image: {file_path}")
                clear_labels_directory(args.temp_dir)
                img_parts = split_image(file_path)
                part_paths = save_image_parts(img_parts, args.temp_dir)
                process_image_parts(part_paths, file_path, args)
    else:
        img = cv2.imread(args.source)
        clear_labels_directory(args.temp_dir)
        img_parts = split_image(args.source)
        part_paths = save_image_parts(img_parts, args.temp_dir)
        process_image_parts(part_paths, args.source, args)

if __name__ == '__main__':
    main()