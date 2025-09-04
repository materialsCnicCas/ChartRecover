from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import os
import mmcv
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
import torch
import json
import time
import shutil
from PIL import Image
from correct_coordinates_bar import correct_coordinates_bar
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from typing import List, Dict

def convert_to_scatter_format(det_data_sample, class_name_list):
    results_labelled = {label: [] for label in class_name_list}
    pred_instances = det_data_sample.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()

    if pred_instances is None or len(pred_instances) == 0:
        return results_labelled
    
    for bbox, score, label in zip(bboxes, scores, labels):
        scatter_entry = list(bbox.astype(float)) + [float(score)]
        if int(label) < len(class_name_list):
            class_name = class_name_list[int(label)]
            results_labelled[class_name].append(scatter_entry)

    for class_name in results_labelled:
        results_labelled[class_name] = sorted(
            results_labelled[class_name],
            key=lambda x: x[-1], 
            reverse=True
        )

    return results_labelled

def create_result_folder(main_path, img_path):
    if os.path.exists(main_path):
        shutil.rmtree(main_path)
    os.makedirs(main_path, exist_ok=True)

    if os.path.isdir(img_path):
        img_files = [f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    else:
        img_files = [os.path.basename(img_path)]
        img_path = os.path.dirname(img_path)
    
    for img_name in img_files:
        result_folder = os.path.join(main_path, os.path.splitext(img_name)[0])
        os.makedirs(result_folder, exist_ok=True)


def Bar_detection(img_path, result_main_folder):
    config_file = './mmdetection/my_configs/codino/codino_bar.py'
    checkpoint_file = './mmdetection/my_checkpoint/codino/bar/codino_bar/checkpoint.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta
    label_list = ["bar"]

    if os.path.isdir(img_path):
        img_files = [f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    else:
        img_files = [os.path.basename(img_path)]
        img_path = os.path.dirname(img_path)
    for img_file in img_files:
        img_file = os.path.join(img_path, img_file)
        img_name = os.path.basename(img_file)
        result_folder_path = os.path.join(result_main_folder, os.path.splitext(img_name)[0])

        if not os.path.exists(f'{result_folder_path}/Bar_result'):
            os.makedirs(f'{result_folder_path}/Bar_result')
        result = inference_detector(model, img_file)
        visualizer.add_datasample(
            'Bar_result',
            mmcv.imread(img_file, channel_order='rgb'),
            data_sample=result,
            out_file=f'{result_folder_path}/Bar_result/Bar_result.jpg',
            draw_gt=False,
            pred_score_thr=0.3
        )
        results_labelled = convert_to_scatter_format(result, label_list)
        
        with open(f"{result_folder_path}/Bar_result/bar_boxes.json", "w") as f:
            json.dump(results_labelled, f)

def Element_detection(img_path, result_main_folder):
    config_file = './mmdetection/my_configs/codino/codino_element.py'
    checkpoint_file = './mmdetection/my_checkpoint/codino/element/codino_element/checkpoint.pth'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = model.dataset_meta
    label_list = [
        "x_title",
        "y_title",
        "plot_area",
        "other",
        "xlabel",
        "ylabel",
        "chart_title",
        "x_tick",
        "y_tick",
        "legend_patch",
        "legend_label",
        "legend_title",
        "legend_area",
        "mark_label",
        "value_label",
        "y_axis_area",
        "x_axis_area",
        "tick_grouping",
    ]
    
    if os.path.isdir(img_path):
        img_files = [f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    else:
        img_files = [os.path.basename(img_path)]
        img_path = os.path.dirname(img_path)
    for img_file in img_files:
        img_file = os.path.join(img_path, img_file)
        img_name = os.path.basename(img_file)
        result_folder_path = os.path.join(result_main_folder, os.path.splitext(img_name)[0])
        shutil.copy(img_file, os.path.join(result_folder_path, 'origin_image.jpg'))

        if not os.path.exists(f'{result_folder_path}/Element_result'):
            os.makedirs(f'{result_folder_path}/Element_result')
        result = inference_detector(model, img_file)
        visualizer.add_datasample(
            'Element_result',
            mmcv.imread(img_file, channel_order='rgb'),
            data_sample=result,
            out_file=f'{result_folder_path}/Element_result/Element_detection_result.jpg',
            draw_gt=False,
            pred_score_thr=0.3
        )

        results_labelled = convert_to_scatter_format(result, label_list)

        with open(f"{result_folder_path}/Element_result/bounding_boxes.json", "w") as f:
            json.dump(results_labelled, f)

        bounding_boxes = results_labelled

        confidence_threshold = 0.3

        image = cv2.imread(img_file)

        label_coordinates = {}

        plot_areas = sorted(
            bounding_boxes["plot_area"], key=lambda el: el[4], reverse=True
        )

        highest_conf_pa = plot_areas[0]

        label_coordinates["plot_area"] = highest_conf_pa

        # Crop bounding boxes 
        legend_coordinates = {}
        cropped_x_labels = crop_bounding_boxes(
            image, bounding_boxes["xlabel"], initial_threshold=confidence_threshold
        )

        cropped_y_labels = crop_bounding_boxes(
            image, bounding_boxes["ylabel"], initial_threshold=confidence_threshold
        )

        cropped_legend_labels = crop_bounding_boxes(
            image, bounding_boxes["legend_label"], initial_threshold=0.7
        )

        cropped_legend_patches = crop_bounding_boxes(
            image, bounding_boxes["legend_patch"][:len(cropped_legend_labels)], initial_threshold=0.0, min_threshold=0.0
        )

        # Save cropped images
        for i, cropped_image in enumerate(cropped_x_labels):
            path = f'{result_folder_path}/Element_result/cropped_xlabels_{i}.png'
            cv2.imwrite(
                path,
                cropped_image,
            )

            label_coordinates[path] = bounding_boxes["xlabel"][i]

        for i, cropped_image in enumerate(cropped_y_labels):
            path = f'{result_folder_path}/Element_result/cropped_ylabels_{i}.png'
            cv2.imwrite(
                path,
                cropped_image,
            )
            label_coordinates[path] = bounding_boxes["ylabel"][i]

        for i, cropped_image in enumerate(cropped_legend_patches):
            path = f'{result_folder_path}/Element_result/cropped_legend_patch_{i}.png'
            cv2.imwrite(
                path,
                cropped_image,
            )
            legend_coordinates[path] = bounding_boxes["legend_patch"][i]
        
        for i, cropped_image in enumerate(cropped_legend_labels):
            path = f'{result_folder_path}/Element_result/cropped_legend_label_{i}.png'
            cv2.imwrite(
                path,
                cropped_image,
            )
            legend_coordinates[path] = bounding_boxes["legend_label"][i]

        with open(f"{result_folder_path}/Element_result/label_coordinates.json", "w") as f:
            json.dump(label_coordinates, f, indent=4)

        with open(f"{result_folder_path}/Element_result/legend_coordinates.json", "w") as f:
            json.dump(legend_coordinates, f, indent=4)

# Function to crop bounding boxes
def crop_bbox(image, boxes, threshold):
    cropped_images = []
    for box in boxes:
        x1, y1, x2, y2, confidence = box
        if confidence >= threshold:
            cropped_image = image[int(y1) : int(y2), int(x1) : int(x2)]
            cropped_images.append(cropped_image)
    return cropped_images

def crop_bounding_boxes(image, boxes, initial_threshold=0.5, min_threshold=0.2, step=0.1):
    boxes = sorted(boxes, key=lambda box: box[4], reverse=True)

    threshold = initial_threshold
    while threshold >= min_threshold:
        cropped_images = crop_bbox(image, boxes, threshold)
        if cropped_images:
            return cropped_images
        threshold -= step
    return []

def TrOCR(img_path, result_main_folder):
    time_loadOCR_1 = time.time()
    ocr_model_path = './ocr_model'
    TrOCR_processor = TrOCRProcessor.from_pretrained(ocr_model_path)
    TrOCR_model = VisionEncoderDecoderModel.from_pretrained(ocr_model_path).to(device)
    # TrOCR_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    # TrOCR_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(device)
    # TrOCR_processor.save_pretrained(ocr_model_path)
    # TrOCR_model.save_pretrained(ocr_model_path)
    time_loadOCR_2 = time.time()
    print("Time to load OCR model:", time_loadOCR_2 - time_loadOCR_1)

    if os.path.isdir(img_path):
        img_files = [f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    else:
        img_files = [os.path.basename(img_path)]
        img_path = os.path.dirname(img_path)
    for img_file in img_files:
        img_file = os.path.join(img_path, img_file)
        img_name = os.path.basename(img_file)
        result_folder_path = os.path.join(result_main_folder, os.path.splitext(img_name)[0])
        print(result_folder_path)
        axis_label_images = []
        axis_label_texts = {}

        legend_label_images = []
        legend_label_texts = {}
        for label_img in os.listdir(result_folder_path + '/Element_result'):
            if "labels" in label_img and ".json" not in label_img:
                axis_label_images.append(f"{result_folder_path}/Element_result/{label_img}")
            if "legend_label" in label_img and ".json" not in label_img:
                legend_label_images.append(f"{result_folder_path}/Element_result/{label_img}")

        for img in axis_label_images:
            image = Image.open(img).convert("RGB")
            pixel_values = TrOCR_processor(image, return_tensors="pt").pixel_values.to(device)
            generated_ids = TrOCR_model.generate(pixel_values)
            generated_text = TrOCR_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
            axis_label_texts[img] = generated_text
        
        for img in legend_label_images:
            image = Image.open(img).convert("RGB")
            pixel_values = TrOCR_processor(image, return_tensors="pt").pixel_values.to(device)
            generated_ids = TrOCR_model.generate(pixel_values)
            generated_text = TrOCR_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]
            legend_label_texts[img] = generated_text
        # Save the results to JSON files
        with open(f'{result_folder_path}/axis_label_texts.json', "w") as f:
            json.dump(axis_label_texts, f, indent=4)
        with open(f'{result_folder_path}/legend_label_texts.json', "w") as f:
            json.dump(legend_label_texts, f, indent=4)
        correct_coordinates_bar(result_folder_path)


if __name__ == "__main__":
    time1 = time.time()

    main_path = './test/test_result/horizontal_bar'
    img_path = './test/test_data/horizontal_bar'

    # main_path = './test/test_result/vertical_bar'
    # img_path = './test/test_data/vertical_bar'

    create_result_folder(main_path, img_path)

    time2 = time.time()
    Bar_detection(img_path, main_path)
    time3 = time.time()
    Element_detection(img_path, main_path)
    time4 = time.time()
    TrOCR(img_path, main_path)
    time5 = time.time()

    print("Bar detection Time cost:", time3 - time2)
    print("Element detection Time cost:", time4 - time3)
    print("TrOCR Time cost:", time5 - time4)    
    print("All Time cost:", time5 - time1)
    
