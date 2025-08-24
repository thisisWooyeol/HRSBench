import argparse
import json
import pickle
import warnings
from typing import Any


def load_gt(jsonl_path: str) -> list[dict[str, Any]]:
    """
    Load ground truth data from JSONL file.
    
    Example entry:
    {
        'prompt': 'a horse below a car.',
        'phrases': ['horse', 'car'],
        'bounding_boxes': [[0.234375, 0.5859375, 0.703125, 0.9375], [0.21484375, 0.1953125, 0.7421875, 0.546875]],
        'num_objects': 2, 'num_bboxes': 2,
        'expected_obj1': 'horse', 'expected_obj2': 'car', 'expected_obj3': '', 'expected_obj4': '',
        'relation1': 'below', 'relation2': '',
        'level': 1
    }
    """
    gt_data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            gt_data.append(json.loads(line))
    return gt_data


def load_pred(pkl_pth: str) -> dict[str, dict[int, list[Any]]]:
    """
    Load predictions from pickle file.
    
    Example entry:
    data["0"] = {
        0: [array(['44.324688', '79.604744', '438.88803', '287.82706', 'airplane'], dtype='<U32')], 
        1: [array(['179.65254', '248.52257', '331.5779', '375.03128', 'car'], dtype='<U32')]
    }
    """
    with open(pkl_pth, "rb") as f:
        pred_data = pickle.load(f)
    return pred_data


def convert_pred_format(pred_data: dict[str, dict[int, list[Any]]]) -> dict[str, dict[int, dict[str, Any]]]:
    """
    Convert prediction data to the format expected by the original functions.
    """
    converted_pred = {}
    for img_id, v in pred_data.items():
        temp_dict = {}
        for obj_id, v2 in v.items():
            if isinstance(v2, list) and len(v2) > 0:
                item = v2[0]  # remove duplicate objects
                temp_dict[obj_id] = {"cls": item[-1]}
                # convert coordinates to float instead of str:
                coords = [float(coord) for coord in item[:4]]
                temp_dict[obj_id]["cords"] = coords  # (xmin, ymin, xmax, ymax)  # origin = top left
        converted_pred[img_id] = temp_dict
    return converted_pred


def convert_gt_format(gt_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert JSONL ground truth format to the format expected by cal_acc function.
    """
    converted_gt = []
    for sample in gt_data:
        # Extract objects (filter out empty strings)
        objs = []
        for i in range(1, 5):  # obj1 to obj4
            obj_key = f"expected_obj{i}"
            if obj_key in sample and sample[obj_key] and sample[obj_key].strip():
                objs.append(sample[obj_key])
        
        # Extract relations (filter out empty strings)
        relations = []
        for i in range(1, 3):  # relation1 to relation2
            rel_key = f"relation{i}"
            if rel_key in sample and sample[rel_key] and sample[rel_key].strip():
                relations.append(sample[rel_key])
        
        converted_gt.append({
            "objs": objs,
            "relations": relations,
            "level": sample.get("level", 0)
        })
    
    return converted_gt


def _check_right(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if xmax1 > xmax2:
        return True
    else:
        return False


def _check_left(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if xmin1 < xmin2:
        return True
    else:
        return False


def _check_above(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if (ymin1 < ymin2) or (ymax1 < ymax2):
        return True
    else:
        return False


def _check_below(obj_1, obj_2):
    """
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
    """
    xmin1, ymin1, xmax1, ymax1 = obj_1
    xmin2, ymin2, xmax2, ymax2 = obj_2
    if (ymax1 > ymax2) or (ymin1 > ymin2):
        return True
    else:
        return False


def _check_between(obj_1, obj_2, obj_3):
    """
    Check if obj1 in between of obj2 and obj3
        obj_1: coordinates of object 1 (xmin, ymin, xmax, ymax)
        obj_2: coordinates of object 2 (xmin, ymin, xmax, ymax)
        obj_3: coordinates of object 3 (xmin, ymin, xmax, ymax)
    """
    # check horizontal dimension:
    if _check_right(obj_1, obj_2) and _check_left(obj_1, obj_3):
        return True
    elif _check_left(obj_1, obj_2) and _check_right(obj_1, obj_3):
        return True
    # check vertical dimension:
    elif _check_below(obj_1, obj_2) and _check_above(obj_1, obj_3):
        return True
    elif _check_above(obj_1, obj_2) and _check_below(obj_1, obj_3):
        return True
    else:
        return False


def _sort_pred_obj(pred_objs: dict[int, dict[str, Any]], gt_objs: list[str]) -> dict[int, dict[str, Any]]:
    """
    Sort the predicted objects based on the GT objects.
    pred_objs: dict of pred objs. key --> obj_id. val --> cls and cords.
    gt_objs: list of gt cls names.
    """
    sorted_pred_objs = {}
    for key, pred_obj in pred_objs.items():
        if pred_obj['cls'] in gt_objs:
            sorted_pred_objs[gt_objs.index(pred_obj['cls'])] = pred_obj
    return sorted_pred_objs


def cal_acc(gt_objs: list[dict[str, Any]], pred_objs: dict[str, dict[int, dict[str, Any]]], level: int) -> float:
    """
    Calculate accuracy for spatial relation task.
    """
    above_spatial_words = ["on", "above", "over", "top"]
    below_spatial_words = ["below", "beneath", "under", "underneath"]
    true_count = 0
    total_count = 0
    
    for img_id, sample in enumerate(gt_objs):
        # Skip if not the correct level
        if sample.get("level", 0) != level:
            continue
            
        total_count += 1
        img_id_str = str(img_id)
        miss_flag = False
        
        # Get the whole predicted classes in this image:
        if img_id_str not in pred_objs.keys():
            continue
            
        pred_cls = [pred_objs[img_id_str][obj_id]['cls'] for obj_id in pred_objs[img_id_str].keys()]

        # Check whether the image contains the correct classes or not:
        for obj_cls in sample['objs']:
            if obj_cls in pred_cls:
                continue
            else:
                miss_flag = True
                break
        if miss_flag:
            continue

        # Sorting the predicted objects based on the GT objects
        sorted_pred_objs = _sort_pred_obj(pred_objs[img_id_str], sample['objs'])
        
        # Determine the hardness level based on the number of objects:
        if len(sample['objs']) == 2:
            # Easy level:
            if sample['relations'][0] == "on the right of" or sample['relations'][0] == "right":
                if _check_right(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    true_count += 1
            elif sample['relations'][0] == "on the left of" or sample['relations'][0] == "left":
                if _check_left(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    true_count += 1
            elif sample['relations'][0] in above_spatial_words:
                if _check_above(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    true_count += 1
            elif sample['relations'][0] in below_spatial_words:
                if _check_below(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                    true_count += 1
        elif len(sample['objs']) == 3:
            # Medium level:
            if len(sample['relations']) == 2:  # normal relation
                # Check first relation:
                if sample['relations'][0] == "on the right of" or sample['relations'][0] == "right":
                    if not _check_right(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                        continue
                elif sample['relations'][0] == "on the left of" or sample['relations'][0] == "left":
                    if not _check_left(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                        continue
                elif sample['relations'][0] in above_spatial_words:
                    if not _check_above(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                        continue
                elif sample['relations'][0] in below_spatial_words:
                    if not _check_below(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords']):
                        continue

                # Check second relation:
                if sample['relations'][1] == "on the right of":
                    if _check_right(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']):
                        true_count += 1
                elif sample['relations'][1] == "on the left of":
                    if _check_left(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']):
                        true_count += 1
                elif sample['relations'][1] in above_spatial_words:
                    if _check_above(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']):
                        true_count += 1
                elif sample['relations'][1] in below_spatial_words:
                    if _check_below(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']):
                        true_count += 1

            else:  # between relation
                if _check_between(sorted_pred_objs[0]['cords'], sorted_pred_objs[1]['cords'],
                                  sorted_pred_objs[2]['cords']):
                    true_count += 1

        elif len(sample['objs']) == 4:
            # Hard level:
            if len(sample['relations']) == 2:  # normal relation
                # Check first relation:
                if sample['relations'][0] == "on the right of":
                    if not (_check_right(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']) and
                            (_check_right(sorted_pred_objs[1]['cords'], sorted_pred_objs[2]['cords']))):
                        continue
                elif sample['relations'][0] == "on the left of":
                    if not (_check_left(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']) and
                            (_check_left(sorted_pred_objs[1]['cords'], sorted_pred_objs[2]['cords']))):
                        continue
                elif sample['relations'][0] in above_spatial_words:
                    if not (_check_above(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']) and
                            (_check_above(sorted_pred_objs[1]['cords'], sorted_pred_objs[2]['cords']))):
                        continue
                elif sample['relations'][0] in below_spatial_words:
                    if not (_check_below(sorted_pred_objs[0]['cords'], sorted_pred_objs[2]['cords']) and
                            (_check_below(sorted_pred_objs[1]['cords'], sorted_pred_objs[2]['cords']))):
                        continue

                # Check second relation:
                if sample['relations'][1] == "on the right of":
                    if _check_right(sorted_pred_objs[0]['cords'], sorted_pred_objs[3]['cords']) and \
                            _check_right(sorted_pred_objs[1]['cords'], sorted_pred_objs[3]['cords']):
                        true_count += 1
                elif sample['relations'][1] == "on the left of":
                    if _check_left(sorted_pred_objs[0]['cords'], sorted_pred_objs[3]['cords']) and \
                            _check_left(sorted_pred_objs[1]['cords'], sorted_pred_objs[3]['cords']):
                        true_count += 1
                elif sample['relations'][1] in above_spatial_words:
                    if _check_above(sorted_pred_objs[0]['cords'], sorted_pred_objs[3]['cords']) and \
                            _check_above(sorted_pred_objs[1]['cords'], sorted_pred_objs[3]['cords']):
                        true_count += 1
                elif sample['relations'][1] in below_spatial_words:
                    if _check_below(sorted_pred_objs[0]['cords'], sorted_pred_objs[3]['cords']) and \
                            _check_below(sorted_pred_objs[1]['cords'], sorted_pred_objs[3]['cords']):
                        true_count += 1

            else:  # between relation
                if _check_between(sorted_pred_objs[0]['cords'],
                                  sorted_pred_objs[2]['cords'],
                                  sorted_pred_objs[3]['cords']) \
                        and _check_between(sorted_pred_objs[1]['cords'],
                                           sorted_pred_objs[2]['cords'],
                                           sorted_pred_objs[3]['cords']):
                    true_count += 1
        else:
            warnings.warn(f"Unexpected number of objects: {len(sample['objs'])}")

    acc = 100 * (true_count / total_count) if total_count > 0 else 0.0
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate spatial relation accuracy.")
    parser.add_argument(
        "--in_pkl_path", 
        type=str, 
        required=True,
        help="Path to the input pickle file."
    )
    parser.add_argument(
        "--gt_jsonl_path", 
        type=str, 
        default="./hrs_dataset/spatial.jsonl",
        help="Path to the ground truth JSONL file.",
    )
    args = parser.parse_args()

    # Load data
    gt_data_raw = load_gt(jsonl_path=args.gt_jsonl_path)
    pred_data_raw = load_pred(pkl_pth=args.in_pkl_path)
    
    # Convert formats
    gt_data = convert_gt_format(gt_data_raw)
    pred_data = convert_pred_format(pred_data_raw)

    NUM_LEVELS = 3
    # Initialize result storage
    acc_per_level = {1: [], 2: [], 3: []}
    all_accuracies = []

    print("Calculating spatial relation accuracy...")
    
    # Calculate accuracy for each level
    for level in range(1, NUM_LEVELS + 1):
        accuracy = cal_acc(gt_data, pred_data, level)
        acc_per_level[level].append(accuracy)
        all_accuracies.append(accuracy)
        
        level_name = ["", "Easy", "Medium", "Hard"][level]
        print(f"{level_name} Level - Accuracy: {accuracy:.2f}%")

    # Calculate overall average
    avg_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0

    # Prepare results dictionary
    all_results = {
        'accuracy_per_level': acc_per_level, 
        'average_accuracy': avg_accuracy
    }
    
    # Save results to JSON file
    result_path = args.in_pkl_path.replace('.pkl', '_results.json')
    with open(result_path, 'w') as f:
        json.dump(all_results, f, sort_keys=True, indent=4)

    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    
    # Print per-level results
    for level in range(1, NUM_LEVELS + 1):
        level_name = ["", "Easy", "Medium", "Hard"][level]
        level_avg_acc = sum(acc_per_level[level]) / len(acc_per_level[level]) if acc_per_level[level] else 0.0
        
        print(f"\n{level_name} Level Results:")
        print(f"  Accuracy: {level_avg_acc:.2f}%")
    
    # Print overall average results
    print("\nOverall Average Results:")
    print(f"  Average Accuracy: {avg_accuracy:.2f}%")
    
    print(f"\nResults saved to: {result_path}")
    print("Done!")
