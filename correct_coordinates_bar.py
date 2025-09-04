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
import statistics
import matplotlib.pyplot as plt

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
                    matched_coords[coords_keys[i]] = list(coords.values())[i]  # 存储原值
                else:
                    matched_coords[coords_keys[i]] = valid_ticks[col_ind[i]]  # 存储匹配值
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

def direction_is(x_coords, y_coords, x_str_flag, y_str_flag):
    x_centers = [(x2,y2) for x1, y1, x2, y2, _ in x_coords.values()]
    y_centers = [(x2,y2) for x1, y1, x2, y2, _ in y_coords.values()]

    if not x_str_flag:
        x_cx = [c[0] for c in x_centers]
        x_cy = [c[1] for c in x_centers]
        x_cx_count = count_close_numbers(x_cx, tolerance=2)
        x_cy_count = count_close_numbers(x_cy, tolerance=2)
        x_type = 'horizontal' if x_cx_count < x_cy_count else 'vertical'
        y_type = 'vertical' if x_type=='horizontal' else 'horizontal'

    if not y_str_flag:
        y_cx = [c[0] for c in y_centers]
        y_cy = [c[1] for c in y_centers]
        y_cx_count = count_close_numbers(y_cx, tolerance=2)
        y_cy_count = count_close_numbers(y_cy, tolerance=2)
        y_type = 'horizontal' if y_cx_count < y_cy_count else 'vertical'
        x_type = 'vertical' if y_type=='horizontal' else 'horizontal'

    else:
        return None, None
    return x_type, y_type

def swap_if_needed(x_coords, y_coords, x_str_flag, y_str_flag):
    x_type, y_type = direction_is(x_coords, y_coords, x_str_flag, y_str_flag)
    if x_type == 'vertical' and y_type == 'horizontal':
        print('x_coords and y_coords are swapped')
        return y_coords, x_coords, 'swap'
    else:
        return x_coords, y_coords, False

def sort_and_check_labels(
    label_coordinates: dict, axis_label_texts: dict, bounding_boxes: dict, bar_boxes: dict, 
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

    x_descending_order = False
    y_descending_order = False

    x_str_flag = False
    y_str_flag = False
    
    xvalues = {}
    yvalues = {}
    for k, v in axis_label_texts.items():
        parsed_value = parse_percentage(v)
        if parsed_value is not None:
            if "ylabel" in k:
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

    x_type, y_type = direction_is(x_coords, y_coords, x_str_flag, y_str_flag)
    if not x_type and not y_type:
        return None, None, None, None, None, None, None
    print('x_type', x_type)
    print('y_type', y_type)

    if x_type == 'vertical' and y_type == 'horizontal':
        direction = 'horizontal'
    elif x_type == 'horizontal' and y_type == 'vertical':
        direction = 'vertical'
    x_str_flag = True

    if direction == 'horizontal':
        all_x2 = [int(box[2]) for box in x_coords.values()]
        baseline_x2 = statistics.mode(all_x2)
        filter_key = [k for k, box in x_coords.items() if abs(int(box[2]) - baseline_x2) > 3]
        x_coords = {k: v for k, v in x_coords.items() if k not in filter_key}
        sorted_xlabels = [item for item in sorted_xlabels if item[0] not in filter_key]

        all_y1 = [int(box[1]) for box in y_coords.values()]
        baseline_y1 = statistics.mode(all_y1)
        filter_key = [k for k, box in y_coords.items() if abs(int(box[1]) - baseline_y1) > 3]
        y_coords = {k: v for k, v in y_coords.items() if k not in filter_key}
        sorted_ylabels = [item for item in sorted_ylabels if item[0] not in filter_key]

    elif direction == 'vertical':
        all_y1 = [int(box[1]) for box in x_coords.values()]
        baseline_y1 = statistics.mode(all_y1)
        filter_key = [k for k, box in x_coords.items() if abs(int(box[1]) - baseline_y1) > 3]
        x_coords = {k: v for k, v in x_coords.items() if k not in filter_key}
        sorted_xlabels = [item for item in sorted_xlabels if item[0] not in filter_key]

        all_x2 = [int(box[2]) for box in y_coords.values()]
        baseline_x2 = statistics.mode(all_x2)
        filter_key = [k for k, box in y_coords.items() if abs(int(box[2]) - baseline_x2) > 3]
        y_coords = {k: v for k, v in y_coords.items() if k not in filter_key}
        sorted_ylabels = [item for item in sorted_ylabels if item[0] not in filter_key]

    x_coords, y_coords = label2tick(x_coords, y_coords, bounding_boxes, x_str_flag, y_str_flag)
    x_coords, y_coords, swap_type = swap_if_needed(x_coords, y_coords, x_str_flag, y_str_flag)

    # Sort the y and x coordinates
    sorted_y_coords = sorted(
        y_coords.items(), key=lambda item: item[1][1], reverse=True
    )  # reverse=True as y direction is top to bottom
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
        return None, None, None, None, None, None, None
    xaggr = {}
    for (k1, v1), (k2, v2) in zip(sorted_x_coords, sorted_xlabels):
        assert k1 == k2
        xaggr[k1] = {"coord": v1, "val": v2}

    yaggr = {}
    for (k1, v1), (k2, v2) in zip(sorted_y_coords, sorted_ylabels):
        assert k1 == k2
        yaggr[k1] = {"coord": v1, "val": v2}
    return {"xs": xaggr, "ys": yaggr}, x_descending_order, y_descending_order, x_str_flag, y_str_flag, direction, swap_type


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
        
    models = {
        "linear": {"slope": None, "intercept": None, "r_sq": -np.inf},
         "linear-x-log-y": {"slope": None, "intercept": None, "r_sq": -np.inf},
    }

    try:
        slope_linear, intercept_linear, _, _, _ = linregress(x_pixels, y_vals)
        y_pred_linear = slope_linear * x_pixels + intercept_linear
        models["linear"]["slope"] = slope_linear
        models["linear"]["intercept"] = intercept_linear
        models["linear"]["r_sq"] = r_squared(y_vals, y_pred_linear)
    except Exception as e:
        print(f"Error in linear fitting: {e}")

    if np.all(y_vals > 0):
        try:
            log_y = np.log10(y_vals)
            slope_linear_logy, intercept_linear_logy, _, _, _ = linregress(x_pixels, log_y)
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

def convert_data_points(conversions, bar_boxes, x_str_flag, y_str_flag, x_descending_order, y_descending_order, main_path, direction, swap_type):
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
    
    def has_close_values_sorted(center_list, tolerance=1):
        if len(center_list) < 2:
            return False
        sorted_list = sorted(center_list)
        for i in range(len(sorted_list) - 1):
            if abs(sorted_list[i] - sorted_list[i + 1]) <= tolerance:
                return True
        return False
    
    visual_data = []
    reality_data = []
    bar_boxes = bar_boxes.get('bar', [])

    if not os.path.exists(f'{main_path}/data_pre'):
        os.makedirs(f'{main_path}/data_pre')

    if not conversions:
        return

    x_labels = {k: v for k, v in conversions["x"].items() if k not in ["slope", "intercept", "type"]} if x_str_flag else {}
    y_labels = {k: v for k, v in conversions["y"].items() if k not in ["slope", "intercept", "type"]} if y_str_flag else {}
    img = cv2.imread(f'{main_path}/origin_image.jpg')
    height, width = img.shape[:2]

    layer_data_flag = False
    x_center_list = [(box[0] + box[2]) / 2 for box in bar_boxes if box[4] > 0.3]
    y_center_list = [(box[1] + box[3]) / 2 for box in bar_boxes if box[4] > 0.3]
    if direction == 'horizontal':
        layer_data_flag = has_close_values_sorted(y_center_list)
    elif direction == 'vertical':
        layer_data_flag = has_close_values_sorted(x_center_list)
    print("layer_data_flag:", layer_data_flag)

    annotations = {}
    marker_number = 1
    if direction == 'horizontal':
        print('Horizontal direction')
        all_x1 = [int(box[0]) for box in bar_boxes if box[4] > 0.3]
        all_x2 = [int(box[2]) for box in bar_boxes if box[4] > 0.3]
        if layer_data_flag:
            bar_baseline = min(all_x1)
        else:
            bar_baseline = statistics.mode(all_x1 + all_x2)
    elif direction == 'vertical':
        print('Vertical direction')
        all_y1 = [int(box[1]) for box in bar_boxes if box[4] > 0.3]
        all_y2 = [int(box[3]) for box in bar_boxes if box[4] > 0.3]
        if layer_data_flag:
            bar_baseline = max(all_y2)
        else:
            bar_baseline = statistics.mode(all_y1 + all_y2)

    for box in bar_boxes:
        mask_flag = False
        [x1, y1, x2, y2, confidence] = box
        if confidence < 0.3:
            continue
        visual_data.append({
            'x0': float(x1),
            'y0': float(y1),
            'width': float(x2 - x1),
            'height': float(y2 - y1)
        })

        if direction == 'horizontal':
            if abs(int(x2) - bar_baseline) <= 2:
                data_x1, data_y1 = x2, (y1 + y2) / 2
                data_x2, data_y2 = x1, (y1 + y2) / 2
            else:
                data_x1, data_y1 = x1, (y1 + y2) / 2
                data_x2, data_y2 = x2, (y1 + y2) / 2
        elif direction == 'vertical':
            if abs(int(y1) - bar_baseline) <= 2:
                data_x1, data_y1 = (x1 + x2) / 2, y1
                data_x2, data_y2 = (x1 + x2) / 2, y2
                mask_flag = True
            else:
                data_x1, data_y1 = (x1 + x2) / 2, y2
                data_x2, data_y2 = (x1 + x2) / 2, y1

        if x_str_flag and x_labels:
            x_label = find_nearest_label(data_y2, x_labels)
            converted_x = str(x_label)
        else:
            if layer_data_flag:
                converted_x = float(convertx(data_x2, conversions["x"]["type"]) - convertx(data_x1, conversions["x"]["type"]))
            else:
                converted_x = float(convertx(data_x2, conversions["x"]["type"]))
        if y_str_flag and y_labels:
            y_label = find_nearest_label(data_y2, y_labels)
            converted_y = str(y_label)
        else:
            if layer_data_flag:
                converted_y = float(converty(data_y2, conversions["y"]["type"]) - converty(data_y1, conversions["y"]["type"]))
            else:
                converted_y = float(converty(data_y2, conversions["y"]["type"]))
        
        if swap_type:
            converted_x, converted_y = converted_y, converted_x

        reality_data.append({
            'x': converted_x,
            'y': converted_y if not np.isnan(converted_y) else None,
        })

        text = f"({converted_x}, {converted_y})"
        (text_width, text_height), baseline = cv2.getTextSize(str(marker_number), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        if direction == 'vertical':
            if mask_flag:
                text_x, text_y = int(x1), min(20, int(data_y2) + 20)
            else:
                text_x, text_y = int(x1), min(height - 20, int(data_y2) - 5)

        elif direction == 'horizontal':
            if mask_flag:
                text_x, text_y = max(20, int(data_x2) -20), int(y2)
            else:
                text_x, text_y = min(width - 20, int(data_x2) + 5) , int(y2)

        cv2.putText(img, str(marker_number), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        annotations[marker_number] = {
            "data": converted_y
        }
        marker_number += 1
    if not os.path.exists(f'{main_path}/converted'):
        os.makedirs(f'{main_path}/converted')
    cv2.imwrite(f'{main_path}/converted/converted.jpg', img)

    with open(f'{main_path}/converted/annotations.json', 'w') as f:
        json.dump(annotations, f, indent=4)

    with open(f'{main_path}/data_pre/visual_data.json', 'w') as f:
        json.dump(visual_data, f, indent=4)
    with open(f'{main_path}/data_pre/reality_data.json', 'w') as f:
        json.dump(reality_data, f, indent=4)

def correct_coordinates_bar(main_path = 'result'):
    with open(f"{main_path}/Element_result/label_coordinates.json", "r") as f:
        label_coordinates = json.load(f)

    with open(f"{main_path}/axis_label_texts.json", "r") as f:
        axis_label_texts = json.load(f)

    with open(f"{main_path}/Element_result/bounding_boxes.json", "r") as f:
        bounding_boxes = json.load(f)
    
    if os.path.exists(f"{main_path}/Bar_result/bar_boxes.json"):
        with open(f"{main_path}/Bar_result/bar_boxes.json", "r") as f:
            bar_boxes = json.load(f)
    else:
        bar_boxes = {}
    
    try:
        coord_val_map, x_descending_order, y_descending_order, x_str_flag, y_str_flag, direction, swap_type = sort_and_check_labels(
            label_coordinates=label_coordinates,
            axis_label_texts=axis_label_texts,
            bounding_boxes=bounding_boxes,
            bar_boxes=bar_boxes,
        )
        conversions = calc_conversion(coord_val_map, x_str_flag, y_str_flag)

        convert_data_points(
            conversions=conversions,
            bar_boxes=bar_boxes,
            x_str_flag=x_str_flag,
            y_str_flag=y_str_flag,
            x_descending_order=x_descending_order,
            y_descending_order=y_descending_order,
            main_path=main_path,
            direction=direction,
            swap_type=swap_type
        )
    except Exception as e:
        print(f"\tCorrecting coordinates did not work!")
        print(e)

if __name__ == "__main__":

    main_path = "./result_vertical_bar/PMC6106759___5_HTML"
    correct_coordinates_bar(main_path)