import argparse
import json
import os
import os.path
from typing import Any
from pathlib import Path

import cv2
import numpy as np


def detect_color_hue_based(hue_value):
    if hue_value < 15:
        color = "red"
    elif hue_value < 22:
        color = "orange"
    elif hue_value < 39:
        color = "yellow"
    elif hue_value < 78:
        color = "green"
    elif hue_value < 131:
        color = "blue"
    else:
        color = "red"

    return color


coco_class_idx = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}


def load_gt(jsonl_pth: str) -> list[dict[str, Any]]:
    """Load ground truth data from JSONL file."""
    gt_list = []
    with open(jsonl_pth, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            # Objects:
            objs = [sample["expected_obj1"], sample["expected_obj2"]]
            for i in range(3, 5):
                if len(sample["expected_obj" + str(i)]) > 0:  # check if there is an object
                    objs.append(sample["expected_obj" + str(i)])

            # Colors:
            colors = [sample["color1"], sample["color2"]]
            for i in range(3, 5):
                if len(sample["color" + str(i)]) > 0:  # check if there is other colors
                    colors.append(sample["color" + str(i)])

            gt_list.append(
                {"prompt": sample["prompt"], "objs": objs, "colors": colors, "level": sample["level"]}
            )

    return gt_list


def load_pred(pred_masks_names, data_len, gt_data):
    img_masks_names_dict = {}
    for idx in range(data_len):
        prompt = gt_data[idx]["prompt"]
        level = gt_data[idx]["level"]
        # img_name = str(idx).zfill(5)+"_"+str(iter_idx).zfill(2)
        img_name = str(idx) + "_" + str(level) + "_" + prompt.replace(" ", "_")
        # import pdb; pdb.set_trace()
        img_masks_names = [
            pred_masks_name
            for pred_masks_name in pred_masks_names
            if img_name in pred_masks_name
        ]
        img_masks_names_dict[img_name] = img_masks_names

    return img_masks_names_dict


def cal_acc(
    gt_data, img_masks_names_dict, level, t2i_out_dir, in_masks_folder
):
    true_counter = 0
    total_num_objs = 0

    for idx, gt_entry in enumerate(gt_data):
        if gt_entry["level"] != level:
            continue

        gt_objs = gt_entry["objs"]
        total_num_objs += len(gt_objs)
        gt_colors = gt_entry["colors"]
        prompt = gt_entry["prompt"]
        img_name = str(idx) + "_" + str(level) + "_" + prompt.replace(" ", "_")
        img = cv2.imread(os.path.join(t2i_out_dir, img_name) + ".jpg")
        if img is None:
            print(f"Warning: Could not load image {img_name}.jpg")
            continue
        
        # Debug: check if we have mask matches
        img_masks_names_per_sample = img_masks_names_dict.get(img_name, [])
        if not img_masks_names_per_sample:
            print(f"Warning: No masks found for image {img_name}")
            continue
        
        print(f"Processing sample {idx}: {len(gt_objs)} objects, {len(img_masks_names_per_sample)} masks")
        
        hsv_frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_frame = hsv_frame[:, :, 0]
        # import pdb; pdb.set_trace()
        for obj_idx in range(len(gt_objs)):  # loop on GT objs
            # 1) make sure the classes are correct:
            gt_obj_id = coco_class_idx[gt_objs[obj_idx]]
            print(f"  Looking for {gt_objs[obj_idx]} (class {gt_obj_id})")
            
            img_masks_name_per_class = []
            detected_classes = set()
            for img_masks_name in img_masks_names_per_sample:
                try:
                    detected_class_id = int(img_masks_name.split("_")[-1].split(".")[0])
                    detected_classes.add(detected_class_id)
                    if detected_class_id == gt_obj_id:
                        img_masks_name_per_class.append(img_masks_name)
                except (ValueError, IndexError):
                    print(f"  Warning: Could not parse class ID from {img_masks_name}")
            
            print(f"  Detected classes: {sorted(detected_classes)}")
            print(f"  Matching masks for {gt_objs[obj_idx]}: {len(img_masks_name_per_class)}")
            
            if len(img_masks_name_per_class):
                # found some predictions match GT class
                # 2) make sure the color is correct:
                for i in range(len(img_masks_name_per_class)):
                    mask = cv2.imread(
                        os.path.join(in_masks_folder, img_masks_name_per_class[i]),
                        cv2.IMREAD_GRAYSCALE,
                    )
                    if mask is None:
                        print(f"Warning: Could not load mask {img_masks_name_per_class[i]}")
                        continue
                    mask = mask / 255.0
                    mask = mask.astype(np.uint8)  # [0->1]
                    hsv_frame_masked = np.multiply(hsv_frame, mask)
                    avg_hue = hsv_frame_masked.sum() / np.count_nonzero(
                        hsv_frame_masked
                    )  # average hue component
                    detected_color = detect_color_hue_based(avg_hue)
                    if detected_color == gt_colors[obj_idx]:
                        true_counter += 1
                        break
        print(true_counter, "/", total_num_objs)
    return 100 * true_counter / total_num_objs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate color classification accuracy using hue-based classifier"
    )
    parser.add_argument(
        "--input_image_dir", 
        type=str, 
        required=True,
        help="Path to the generated images directory"
    )
    
    parser.add_argument(
        "--input_mask_dir", 
        type=str, 
        required=True,
        help="Path to the mask directory"
    )
    parser.add_argument(
        "--gt_jsonl_path", 
        type=str, 
        default="./hrs_dataset/color.jsonl",
        help="Path to ground truth JSONL file"
    )
    args = parser.parse_args()
    
    # Load GT:
    gt_data = load_gt(jsonl_pth=args.gt_jsonl_path)
    pred_masks_names = os.listdir(args.input_mask_dir)

    # Load Predictions:
    img_masks_names_dict = load_pred(
        pred_masks_names=pred_masks_names,
        data_len=len(gt_data),
        gt_data=gt_data,
    )

    NUM_LEVEL = 3
    avg_acc = []
    acc_per_level = {1: [], 2: [], 3: []}
    for level in range(1, NUM_LEVEL + 1):
        # Calculate the counting Accuracy:
        acc = cal_acc(
            gt_data,
            img_masks_names_dict,
            level=level,
            t2i_out_dir=args.input_image_dir,
            in_masks_folder=args.input_mask_dir,
        )
        avg_acc.append(acc)
        acc_per_level[level].append(acc)

        # Print iteration results for this level
        level_name = ["", "Easy", "Medium", "Hard"][level]
        print(f"{level_name} Level - Accuracy: {acc:.2f} %")

    # Prepare results dictionary
    all_results = {"acc": acc_per_level, "avg": sum(avg_acc) / len(avg_acc)}
    
    # Save results to JSON file
    result_dir = Path(args.input_mask_dir).resolve().parent
    result_path = os.path.join(result_dir, "color_results.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, sort_keys=True, indent=4)

    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)    

    # Print per-level average results
    for level in range(1, NUM_LEVEL + 1):
        level_name = ["", "Easy", "Medium", "Hard"][level]

        acc = acc_per_level[level][0] if acc_per_level[level] else 0.0
        print(f"\n {level_name} Level - Accuracy: {acc:.2f} %")
    
    # Print overall average results
    print(f"\n Overall Average - Accuracy: {sum(avg_acc) / len(avg_acc):.2f} %")

    print(f"\nResults saved to {result_path}")
    print("Done!")
