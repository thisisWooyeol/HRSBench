import argparse
import json
import pickle
import warnings
from typing import Any

from hrsbench import HRSBENCH_ROOT

def load_gt(jsonl_path: str) -> list[dict[str, Any]]:
    """
    Example entry:
    {
        'prompt': 'two cups filled with steaming hot coffee sit side-by-side on a wooden table.', 
        'phrases': ['cup', 'cup', 'table', 'steam', 'steam'], 
        'bounding_boxes': [[0.25, 0.390625, 0.47265625, 0.5625], [0.52734375, 0.390625, 0.75, 0.5625], [0.0390625, 0.625, 0.9609375, 1.0], [0.361328125, 0.3359375, 0.419921875, 0.390625], [0.638671875, 0.3359375, 0.697265625, 0.390625]], 
        'num_objects': 5, 'num_bboxes': 5, 
        'vanilla_prompt': '2 cup', 
        'expected_n1': 2, 'expected_obj1': 'cup', 
        'expected_n2': 0, 'expected_obj2': '', 
        'level': 1
    }
    """
    gt_data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            gt_data.append(json.loads(line))
    return gt_data


def load_pred(pkl_pth) -> dict[str, dict[int, list[Any]]]:
    """
    Example entry:
    data["0"] = {
        0: [array(['44.324688', '79.604744', '438.88803', '287.82706', 'airplane'],dtype='<U32')], 
        1: [array(['179.65254', '248.52257', '331.5779', '375.03128', 'car'],dtype='<U32')]
    }
    """
    with open(pkl_pth, "rb") as f:
        pred_data = pickle.load(f)
    return pred_data


def compare_entry(gt_entry: dict[str, Any], pred_entry: dict[int, list[Any]]) -> tuple[int, int, int]:
    """
    Compare the ground truth entry with the predicted entry.

    Returns:
    tuple[int, int, int]: A tuple containing the number of true positives, false positives, and false negatives.
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    obj1 = gt_entry["expected_obj1"]
    n1 = gt_entry["expected_n1"]
    obj2 = gt_entry["expected_obj2"] if gt_entry["expected_obj2"] else None
    n2 = gt_entry["expected_n2"]

    obj1_pred = obj2_pred = 0
    for pred in pred_entry.values():
        if pred[0][-1] == obj1:
            obj1_pred += 1
        elif obj2 and pred[0][-1] == obj2:
            obj2_pred += 1

    # Calculate metrics for object 1
    true_pos += min(n1, obj1_pred)
    false_pos += max(0, obj1_pred - n1)
    false_neg += max(0, n1 - obj1_pred)

    # Calculate metrics for object 2 (if exists)
    if obj2 and n2 > 0:
        true_pos += min(n2, obj2_pred)
        false_pos += max(0, obj2_pred - n2)
        false_neg += max(0, n2 - obj2_pred)

    return true_pos, false_pos, false_neg

def calc_accuracy(gt_data: list[dict[str, Any]], pred_data: dict[str, dict[int, list[Any]]], level : int) -> tuple[float, float]:
    """
    Calculate precision and recall based on ground truth and predicted data. Only consider objects at the given level.

    Returns:
        tuple[float, float]: A tuple containing precision and recall.
    """
    total_true_pos = 0
    total_false_pos = 0
    total_false_neg = 0

    for idx, gt_entry in enumerate(gt_data):
        img_id = str(idx)
        if gt_entry['level'] != level:
            continue
        if img_id in pred_data:
            pred_entry = pred_data[img_id]
            true_pos, false_pos, false_neg = compare_entry(gt_entry, pred_entry)
            total_true_pos += true_pos
            total_false_pos += false_pos
            total_false_neg += false_neg
        else:
            warnings.warn(f"Image ID {img_id} not found in predictions. Please check the input data.")

    precision = total_true_pos / (total_true_pos + total_false_pos) if (total_true_pos + total_false_pos) > 0 else 0.0
    recall = total_true_pos / (total_true_pos + total_false_neg) if (total_true_pos + total_false_neg) > 0 else 0.0

    return precision, recall

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate counting accuracy.")
    parser.add_argument(
        "--in_pkl_path", 
        type=str, 
        required=True,
        help="Path to the input pickle file."
    )
    parser.add_argument(
        "--gt_jsonl_path", 
        type=str, 
        default=f"{HRSBENCH_ROOT}/hrs_dataset/counting.jsonl",
        help="Path to the ground truth JSONL file.",
    )
    args = parser.parse_args()

    gt_data = load_gt(jsonl_path=args.gt_jsonl_path)
    pred_data = load_pred(pkl_pth=args.in_pkl_path)

    NUM_LEVEL = 3
    # Initialize result storage
    precisions_per_level = {1: [], 2: [], 3: []}
    recalls_per_level = {1: [], 2: [], 3: []}
    f1_per_level = {1: [], 2: [], 3: []}
    all_precisions = []
    all_recalls = []
    all_f1 = []

    # Calculate accuracy for each level
    for level in range(1, NUM_LEVEL + 1):
        precision, recall = calc_accuracy(gt_data, pred_data, level)
        
        # Convert to percentage
        precision *= 100
        recall *= 100
        
        # Calculate F1 score
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        # Store results
        precisions_per_level[level].append(precision)
        recalls_per_level[level].append(recall)
        f1_per_level[level].append(f1)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1.append(f1)
        
        # Print iteration results for this level
        level_name = ["", "Easy", "Medium", "Hard"][level]
        print(f"{level_name} Level - Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}%")

    # Prepare results dictionary
    avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0.0
    avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
    avg_f1_score = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    
    all_results = {
        "precisions_per_level": precisions_per_level,
        "recalls_per_level": recalls_per_level,
        "f1_per_level": f1_per_level,
        "average": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1_score,
        },
    }

    # Save results to JSON file
    result_path = args.in_pkl_path.replace('.pkl', '_results.json')
    with open(result_path, "w") as f:
        json.dump(all_results, f, sort_keys=True, indent=4)
    
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    # Print per-level average results
    for level in range(1, NUM_LEVEL + 1):
        level_name = ["", "Easy", "Medium", "Hard"][level]
        level_avg_precision = sum(precisions_per_level[level]) / len(precisions_per_level[level]) if precisions_per_level[level] else 0.0
        level_avg_recall = sum(recalls_per_level[level]) / len(recalls_per_level[level]) if recalls_per_level[level] else 0.0
        level_avg_f1 = sum(f1_per_level[level]) / len(f1_per_level[level]) if f1_per_level[level] else 0.0
        
        print(f"\n{level_name} Level Results:")
        print(f"  Precision: {level_avg_precision:.2f}%")
        print(f"  Recall: {level_avg_recall:.2f}%") 
        print(f"  F1 Score: {level_avg_f1:.2f}%")
    
    # Print overall average results
    print("\nOverall Average Results:")
    print(f"  Precision: {avg_precision:.2f}%")
    print(f"  Recall: {avg_recall:.2f}%")
    print(f"  F1 Score: {avg_f1_score:.2f}%")
    
    print(f"\nResults saved to: {result_path}")
    print("Done!")
