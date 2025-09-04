import json
from scipy.stats import linregress
from scipy.optimize import linear_sum_assignment
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import bisect
import random
import matplotlib.pyplot as plt
import statistics

def label2tick(x_coords, y_coords, bounding_boxes, x_str_flag, y_str_flag):
    x_tick_list = bounding_boxes['x_tick']
    y_tick_list = bounding_boxes['y_tick']
    x_tick_vaild = []
    y_tick_vaild = []
    for i, x_tick in enumerate(x_tick_list):
        if x_tick[4] < 0.2:
            continue
        x_tick_vaild.append(x_tick)
    for i, y_tick in enumerate(y_tick_list):
        if y_tick[4] < 0.2:
            continue
        y_tick_vaild.append(y_tick)

    def match_coords_to_ticks(coords, tick_list):
        if not coords or not tick_list:
            return coords.copy()

        valid_ticks = [tick for tick in tick_list if tick[4] >= 0.1]
        if not valid_ticks:
            return coords.copy()

        coords_centers = [((val[0] + val[2]) / 2, (val[1] + val[3]) / 2) for val in coords.values()]
        tick_centers = [((tick[0] + tick[2]) / 2, (tick[1] + tick[3]) / 2) for tick in valid_ticks]

        n, m = len(coords_centers), len(tick_centers)
        distance_matrix = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                dx = coords_centers[i][0] - tick_centers[j][0]
                dy = coords_centers[i][1] - tick_centers[j][1]
                distance_matrix[i][j] = math.sqrt(dx**2 + dy**2)

        row_ind, col_ind = linear_sum_assignment(distance_matrix)
        
        matched_coords = {}
        coords_keys = list(coords.keys())
        for i in range(n):
            if i < len(row_ind) and col_ind[i] < m:
                if valid_ticks[col_ind[i]][4] < 0.5:
                    matched_coords[coords_keys[i]] = list(coords.values())[i]
                else:
                    matched_coords[coords_keys[i]] = valid_ticks[col_ind[i]]
            else:
                matched_coords[coords_keys[i]] = list(coords.values())[i] 

        return matched_coords

    x_coords_update = match_coords_to_ticks(x_coords, bounding_boxes['x_tick'])
    y_coords_update = match_coords_to_ticks(y_coords, bounding_boxes['y_tick'])

    if len(x_tick_vaild) < len(x_coords) or x_str_flag:
        print('x is not enough valid ticks for coordinates')
        x_coords_update =  x_coords
    if len(y_tick_vaild) < len(y_coords) or y_str_flag:
        print('y is not enough valid ticks for coordinates')
        y_coords_update =  y_coords
    return x_coords_update, y_coords_update

def calculate_centers(coords):
    centers = []
    for box in coords.values():
        x1, y1, x2, y2, _ = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))
    return centers

def direction_is(x_coords, y_coords):
    x_centers = calculate_centers(x_coords)
    y_centers = calculate_centers(y_coords)

    x_cx = [c[0] for c in x_centers]
    x_cy = [c[1] for c in x_centers]
    x_cx_count = count_close_numbers(x_cx, tolerance=2)
    x_cy_count = count_close_numbers(x_cy, tolerance=2)
    x_type = 'horizontal' if x_cx_count < x_cy_count else 'vertical'

    y_cx = [c[0] for c in y_centers]
    y_cy = [c[1] for c in y_centers]
    y_cx_count = count_close_numbers(y_cx, tolerance=2)
    y_cy_count = count_close_numbers(y_cy, tolerance=2)
    y_type = 'horizontal' if y_cx_count < y_cy_count else 'vertical'
    return x_type, y_type


def swap_if_needed(x_coords, y_coords):
    x_type, y_type = direction_is(x_coords, y_coords)
    if x_type == 'vertical' and y_type == 'horizontal':
        print('x_coords and y_coords are swapped')
        return y_coords, x_coords, 'swap'
    else:
        return x_coords, y_coords, False

def is_contained(box_a, box_b, threshold=3):
    A_x1, A_y1, A_x2, A_y2 = box_a[:4]
    B_x1, B_y1, B_x2, B_y2 = box_b[:4]
    adjusted_A_x1 = max(0, A_x1 - threshold)
    adjusted_A_y1 = max(0, A_y1 - threshold) 
    adjusted_A_x2 = A_x2 + threshold
    adjusted_A_y2 = A_y2 + threshold
    
    return (B_x1 >= adjusted_A_x1) and (B_y1 >= adjusted_A_y1) and \
           (B_x2 <= adjusted_A_x2) and (B_y2 <= adjusted_A_y2)

def sort_and_check_labels(
    label_coordinates: dict, axis_label_texts: dict, bounding_boxes: dict
):
    def parse_percentage(s):
        if isinstance(s, str):
            s = s.strip()
            if s.endswith('%'):
                try:
                    return float(s[:-1]) / 100
                except ValueError:
                    return None
        try:
            return float(s)
        except ValueError:
            return None
    x_area = bounding_boxes['x_axis_area'][0]  if bounding_boxes['x_axis_area'][0][4] > 0.5 else []
    y_area = bounding_boxes['y_axis_area'][0]  if bounding_boxes['y_axis_area'][0][4] > 0.5 else []

    x_descending_order = False
    y_descending_order = False

    x_str_flag = False
    y_str_flag = False
    
    xvalues = {}
    yvalues = {}

    for k, v in axis_label_texts.items():
        parsed_value = parse_percentage(v)
        if parsed_value is not None:
            if "xlabel" in k:
                xvalues[k] = parsed_value
            elif "ylabel" in k:
                yvalues[k] = parsed_value

    if len(xvalues) == 0:
        xvalues = {k: v for k, v in axis_label_texts.items() if "xlabel" in k}
        x_str_flag = True
        print("x_str_flag is True")
    if len(yvalues) == 0:
        yvalues = {k: v for k, v in axis_label_texts.items() if "ylabel" in k}
        y_str_flag = True
        print("y_str_flag is True")

    sorted_ylabels = sorted(yvalues.items(), key=lambda item: item[1])
    sorted_xlabels = sorted(xvalues.items(), key=lambda item: item[1])

    y_coords = {k: label_coordinates[k] for k, v in yvalues.items()}
    x_coords = {k: label_coordinates[k] for k, v in xvalues.items()}
    thread_area = 5
    delete_keys = []
    for k, box in list(x_coords.items()):
        if x_area and not is_contained(x_area, box):
                delete_keys.append(k)
                del x_coords[k]

    for k, box in list(y_coords.items()):
        if y_area and not is_contained(y_area, box):
                delete_keys.append(k)
                del y_coords[k]

    sorted_xlabels = [item for item in sorted_xlabels if item[0] not in delete_keys]
    sorted_ylabels = [item for item in sorted_ylabels if item[0] not in delete_keys]

    x_coords, y_coords = label2tick(x_coords, y_coords, bounding_boxes, x_str_flag, y_str_flag)
    x_coords, y_coords, swap_type = swap_if_needed(x_coords, y_coords)
    if swap_type:
        xvalues, yvalues = yvalues, xvalues
        sorted_xlabels, sorted_ylabels = sorted_ylabels, sorted_xlabels

    sorted_y_coords = sorted(
        y_coords.items(), key=lambda item: item[1][1], reverse=True
    )
    sorted_x_coords = sorted(x_coords.items(), key=lambda item: item[1][0])

    if swap_type:
        xvalues, yvalues = yvalues, xvalues
        sorted_xlabels, sorted_ylabels = sorted_ylabels, sorted_xlabels
        x_str_flag, y_str_flag = y_str_flag, x_str_flag

    if x_str_flag:
        x_coords_paths = [item[0] for item in sorted_x_coords]
        path_to_index = {path: idx for idx, path in enumerate(x_coords_paths)}
        sorted_xlabels = sorted(sorted_xlabels, key=lambda item: path_to_index[item[0]])

    if y_str_flag:
        y_coords_paths = [item[0] for item in sorted_y_coords]
        path_to_index = {path: idx for idx, path in enumerate(y_coords_paths)}
        sorted_ylabels = sorted(sorted_ylabels, key=lambda item: path_to_index[item[0]])
    # ocr result change
    operations = [
        lambda sl: sorted(sl, key=lambda item: item[1], reverse=True),
        lambda sl: sorted([(label, 9.0 if value == 6.0 else value) for label, value in sl], key=lambda item: item[1]),
        lambda sl: sorted([(label, 6.0 if value == 9.0 else value) for label, value in sl], key=lambda item: item[1]),
        lambda sl: sorted([(label, 9.0 if value == 6.0 else (6.0 if value == 9.0 else value)) for label, value in sl], key=lambda item: item[1]),
        lambda sl: sorted(
            [(label, 9.0 if value == 6.0 else value) for label, value in sl],
            key=lambda item: item[1],
            reverse=True
        ),
        lambda sl: sorted(
            [(label, 6.0 if value == 9.0 else value) for label, value in sl],
            key=lambda item: item[1],
            reverse=True
        ),
        lambda sl: sorted(
            [(label, 9.0 if value == 6.0 else (6.0 if value == 9.0 else value)) for label, value in sl],
            key=lambda item: item[1],
            reverse=True
        ),
    ]

    if [label for label, _ in sorted_xlabels] != [coord for coord, _ in sorted_x_coords] and not x_str_flag:
        for operation in operations:
            temp_sorted = operation(sorted_xlabels)
            temp_labels = [label for label, _ in temp_sorted]
            coord_labels = [coord for coord, _ in sorted_x_coords]
            if temp_labels == coord_labels:
                sorted_xlabels = temp_sorted
                break
    
    if [label for label, _ in sorted_ylabels] != [coord for coord, _ in sorted_y_coords] and not y_str_flag:
        for operation in operations:
            temp_sorted = operation(sorted_ylabels)
            temp_labels = [label for label, _ in temp_sorted]
            coord_labels = [coord for coord, _ in sorted_y_coords]
            if temp_labels == coord_labels:
                sorted_ylabels = temp_sorted
                break

    try:
        assert [label for label, _ in sorted_xlabels] == [
            coord for coord, _ in sorted_x_coords
        ], "The keys of sorted_xlabels and sorted_x_coords are not in the same order."

        assert [label for label, _ in sorted_ylabels] == [
            coord for coord, _ in sorted_y_coords
        ], "The keys of sorted_ylabels and sorted_y_coords are not in the same order."

    except:
        print("Error in sorting coordinates and labels.")
        return None, None, None, None, None

    xaggr = {}
    for (k1, v1), (k2, v2) in zip(sorted_x_coords, sorted_xlabels):
        assert k1 == k2
        xaggr[k1] = {"coord": v1, "val": v2}
    

    yaggr = {}
    for (k1, v1), (k2, v2) in zip(sorted_y_coords, sorted_ylabels):
        assert k1 == k2
        yaggr[k1] = {"coord": v1, "val": v2}

    return {"xs": xaggr, "ys": yaggr}, x_descending_order, y_descending_order, x_str_flag, y_str_flag


def get_best_fit(coord_map: dict, direction="x", is_str_flag=False):
    if is_str_flag:
        str_dict = {}
        for k, v in coord_map.items():
            str_dict[v["val"]] = (v["coord"][0 if direction == "x" else 1] + v['coord'][2 if direction == "x" else 3]) / 2.0
        return str_dict
    
    points = [
        ((v["coord"][0 if direction == "x" else 1] + v['coord'][2 if direction == "x" else 3]) / 2.0,  # Average of x1 and x2 or y1 and y2
         v["val"])
        for k, v in coord_map.items()
    ]
    x_pixels, y_vals = zip(*points)
    x_pixels = np.array(x_pixels)
    y_vals = np.array(y_vals)

    def r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    def exponential_model_base10(x, a, b):
        return np.power(10, a * x + b)

        ratios.sort(key=lambda x: x[0])
        median_idx = len(ratios) // 2
        r_med, med_i, _ = ratios[median_idx]
        scale = r_med
        offset = true_values[med_i] - scale * pixel_values[med_i]
        return float(scale), float(offset)

    models = {
        "linear": {"slope": None, "intercept": None, "r_sq": -np.inf},
         "linear-x-log-y": {"slope": None, "intercept": None, "r_sq": -np.inf},
    }

    try:
        slope_linear, intercept_linear, rvalue, _, _ = linregress(x_pixels, y_vals)
        y_pred_linear = slope_linear * x_pixels + intercept_linear
        models["linear"]["slope"] = slope_linear
        models["linear"]["intercept"] = intercept_linear
        models["linear"]["r_sq"] = r_squared(y_vals, y_pred_linear)

    except Exception as e:
        print(f"Error in linear fitting: {e}")

    if np.all(y_vals > 0):
        try:
            log_y = np.log10(y_vals)
            slope_linear_logy, intercept_linear_logy, rvalue, _, _ = linregress(x_pixels, log_y)
            y_pred_linear_logy = slope_linear_logy * x_pixels + intercept_linear_logy
            models["linear-x-log-y"]["slope"] = slope_linear_logy
            models["linear-x-log-y"]["intercept"] = intercept_linear_logy
            models["linear-x-log-y"]["r_sq"] = r_squared(log_y, y_pred_linear_logy)

        except Exception as e:
            print(f"Error in linear-log fitting: {e}")
    best_model = max(models.items(), key=lambda item: item[1]["r_sq"] if item[1]["r_sq"] != -np.inf else -np.inf)
    best_model_name, best_model_params = best_model
    print(f"{direction} Best model: {best_model_name}")
    return {
        "slope": best_model_params["slope"],
        "intercept": best_model_params["intercept"],
        "type": best_model_name,
    }


def calc_conversion(coord_val_map: dict, x_str_flag, y_str_flag):
    
    try:
        xpix2val = get_best_fit(coord_val_map["xs"], direction="x", is_str_flag=x_str_flag)
        ypix2val = get_best_fit(coord_val_map["ys"], direction="y", is_str_flag=y_str_flag)
    except:
        return None
    return {"x": xpix2val, "y": ypix2val}

def count_close_numbers(lst, tolerance):
    sorted_lst = sorted(lst)
    max_count = 1
    current_count = 1

    for i in range(1, len(sorted_lst)):
        if sorted_lst[i] - sorted_lst[i - 1] <= tolerance:
            current_count += 1
            if current_count > max_count:
                max_count = current_count
        else:
            current_count = 1

    return max_count

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max, _ = box1
    x2_min, y2_min, x2_max, y2_max, _ = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def convert_data_points(conversions, scatter_boxes, bounding_boxes, x_str_flag, y_str_flag, x_descending_order, y_descending_order, main_path = 'result'):
    def convertx(x, model_type):
        if model_type == "linear-x-log-y":
            return 10 ** (x * conversions["x"]["slope"] + conversions["x"]["intercept"])
        elif model_type == "linear":
            return x * conversions["x"]["slope"] + conversions["x"]["intercept"]

    def converty(y, model_type):
        if model_type == "linear-x-log-y":
            return 10 ** (y * conversions["y"]["slope"] + conversions["y"]["intercept"])
        elif model_type == "linear":
            return y * conversions["y"]["slope"] + conversions["y"]["intercept"]
    
    def find_nearest_label(value, label_positions):
        sorted_labels = sorted(label_positions.items(), key=lambda x: x[1])
        positions = [pos for _, pos in sorted_labels]
        labels = [label for label, _ in sorted_labels]
        idx = bisect.bisect_left(positions, value)
        if idx == 0:
            return labels[0]
        if idx == len(positions):
            return labels[-1]
        before = positions[idx - 1]
        after = positions[idx]
        if after - value < value - before:
            return labels[idx]
        else:
            return labels[idx - 1]
    
    def scatter_box_filterd(boxes, initial_threshold=0.3, min_threshold=0.1, step=0.2):
        boxes = sorted(boxes, key=lambda box: box[4], reverse=True)

        threshold = initial_threshold
        while threshold + 1e-8 >= min_threshold:
            box_filter = [box for box in boxes if box[4] >= threshold]
            if len(box_filter) > 0:
                return box_filter
            threshold -= step
        return []
    
    visual_data = []
    reality_data = []
    legend_area = bounding_boxes['legend_area'][0] if bounding_boxes['legend_area'][0][4] > 0.5 else []
    x_area = bounding_boxes['x_axis_area'][0]  if bounding_boxes['x_axis_area'][0][4] > 0.5 else []
    y_area = bounding_boxes['y_axis_area'][0]  if bounding_boxes['y_axis_area'][0][4] > 0.5 else []

    scatter_boxes = scatter_boxes.get('scatter', [])
    scatter_filterd = scatter_box_filterd(scatter_boxes)

    for box in scatter_filterd:
        x1, y1, x2, y2, _ = box
        if legend_area and is_contained(legend_area, box):
                continue
        if x_area and is_contained(x_area, box):
                continue
        if y_area and is_contained(y_area, box):
                continue
        data_x, data_y = [(x1 + x2) / 2, (y1 + y2) / 2]
        visual_data.append({
            'x': float(data_x),
            'y': float(data_y)
        })

    if not os.path.exists(f'{main_path}/data_pre'):
        os.makedirs(f'{main_path}/data_pre')
    with open(f'{main_path}/data_pre/visual_data.json', 'w') as f:
        json.dump(visual_data, f, indent=4)
    if not conversions:
        return

    x_labels = {k: v for k, v in conversions["x"].items() if k not in ["slope", "intercept", "type"]} if x_str_flag else {}
    y_labels = {k: v for k, v in conversions["y"].items() if k not in ["slope", "intercept", "type"]} if y_str_flag else {}

    img = cv2.imread(f'{main_path}/origin_image.jpg')
    height, width = img.shape[:2]

    annotations = {}
    marker_number = 1
    used_text_boxes = []

    def get_text_rect(text, text_x, text_y, font_scale=0.4, thickness=1):
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        rect_x1 = text_x
        rect_y1 = text_y - text_height
        rect_x2 = text_x + text_width
        rect_y2 = text_y + baseline
        return (rect_x1, rect_y1, rect_x2, rect_y2)

    def check_overlap(rect1, rect2):
        if (rect1[0] < rect2[2] and rect1[2] > rect2[0] and
            rect1[1] < rect2[3] and rect1[3] > rect2[1]):
            return True
        return False

    def find_text_position(start_x, start_y, text_width, text_height, img_width, img_height, used_boxes):
        directions = [
            (5, -5),
            (5, 5),
            (-5, -5),
            (-5, 5),
        ]
        
        for direction in directions:
            text_x = start_x + direction[0]
            text_y = start_y + direction[1]

            for step in range(1, 8):
                offset_x = direction[0] * step
                offset_y = direction[1] * step
                text_x = start_x + offset_x
                text_y = start_y + offset_y

                text_rect = get_text_rect(str(marker_number), text_x, text_y)

                if (text_rect[0] < 0 or text_rect[2] > img_width or
                    text_rect[1] < 0 or text_rect[3] > img_height):
                    continue
                overlap = False
                for used_box in used_boxes:
                    if check_overlap(text_rect, used_box):
                        overlap = True
                        break
                
                if not overlap:
                    return text_x, text_y, text_rect

        default_x = start_x + directions[0][0]
        default_y = start_y + directions[0][1]
        return default_x, default_y, get_text_rect(str(marker_number), default_x, default_y)

    for box in scatter_filterd:
        x1, y1, x2, y2, _ = box
        if legend_area and is_contained(legend_area, box):
                continue
        if x_area and is_contained(x_area, box):
                continue
        if y_area and is_contained(y_area, box):
                continue
        data_x, data_y = [(x1 + x2) / 2, (y1 + y2) / 2]
        if x_str_flag and x_labels:
            x_label = find_nearest_label(data_x, x_labels)
            converted_x = x_label
        else:
            converted_x = float(convertx(data_x, conversions["x"]["type"]))
        
        if y_str_flag and y_labels:
            y_label = find_nearest_label(data_y, y_labels)
            converted_y = y_label
        else:
            converted_y = float(converty(data_y, conversions["y"]["type"]))

        reality_data.append({
            'x': converted_x,
            'y': converted_y,
        })

        with open(f'{main_path}/data_pre/reality_data.json', 'w') as f:
            json.dump(reality_data, f, indent=4)

        if x_str_flag and y_str_flag:
            text = f"({converted_x}, {converted_y})"
        elif x_str_flag:
            text = f"({converted_x}, {converted_y:.4f})"
        elif y_str_flag:
            text = f"({converted_x:.4f}, {converted_y})"
        else:
            text = f"({converted_x:.4f}, {converted_y:.4f})"

        (text_width, text_height), baseline = cv2.getTextSize(
                    str(marker_number), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        text_x, text_y, text_rect = find_text_position(
            int(data_x), 
            int(data_y),
            text_width,
            text_height,
            width,
            height,
            used_text_boxes
        )
        used_text_boxes.append(text_rect)

        start_pt = (int(data_x), int(data_y))
        end_pt   = (text_x + text_width // 2, text_y - text_height // 2)

        cv2.line(img,
            start_pt,
            end_pt,
            color=(255, 165, 0),
            thickness=1,
            lineType=cv2.LINE_AA)

        cv2.putText(img, str(marker_number), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)


        if x_str_flag:
            annotations[marker_number] = {
                "data": converted_y if y_str_flag else converted_y
            }
        else:
            annotations[marker_number] = {
                "x": converted_x if x_str_flag else converted_x,
                "y": converted_y if y_str_flag else converted_y
            }
        marker_number += 1
    if not os.path.exists(f'{main_path}/converted'):
        os.makedirs(f'{main_path}/converted')
    cv2.imwrite(f'{main_path}/converted/converted.jpg', img)
    with open(f'{main_path}/converted/annotations.json', 'w') as f:
        json.dump(annotations, f, indent=4)


def correct_coordinates_scatter(main_path = 'result'):
    with open(f"{main_path}/Element_result/label_coordinates.json", "r") as f:
        label_coordinates = json.load(f)

    with open(f"{main_path}/axis_label_texts.json", "r") as f:
        axis_label_texts = json.load(f)

    with open(f"{main_path}/Element_result/bounding_boxes.json", "r") as f:
        bounding_boxes = json.load(f)
    
    if os.path.exists(f"{main_path}/Scatter_result/scatter_boxes.json"):
        with open(f"{main_path}/Scatter_result/scatter_boxes.json", "r") as f:
            scatter_boxes = json.load(f)
    else:
        scatter_boxes = {}

    
    try:
        coord_val_map, x_descending_order, y_descending_order, x_str_flag, y_str_flag = sort_and_check_labels(
            label_coordinates=label_coordinates,
            axis_label_texts=axis_label_texts,
            bounding_boxes=bounding_boxes,
        )
        conversions = calc_conversion(coord_val_map, x_str_flag, y_str_flag)

        convert_data_points(
            conversions=conversions,
            scatter_boxes=scatter_boxes,
            bounding_boxes=bounding_boxes,
            x_str_flag=x_str_flag,
            y_str_flag=y_str_flag,
            x_descending_order=x_descending_order,
            y_descending_order=y_descending_order,
            main_path = main_path
        )
    except Exception as e:
        print(f"\tCorrecting coordinates did not work!")
        print(e)

if __name__ == "__main__":

    main_path = "./test_result/co-dino/scatter/PMC2705787___g007"
    correct_coordinates_scatter(main_path)