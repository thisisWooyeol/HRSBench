import pickle
import csv
import json
from collections import Counter


def load_box(pickle_file: str) -> dict[str, list]:
    """Load pickle file and return the data dictionary."""
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data


def load_color_prompts(csv_file: str) -> dict[str, dict]:
    """Load the color prompts CSV and return a dictionary with meta_prompt as key."""
    prompts_data = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            meta_prompt = row['meta_prompt'].strip()
            prompts_data[meta_prompt] = {
                'obj1': row['obj1'].strip(),
                'obj2': row['obj2'].strip(),
                'obj3': row['obj3'].strip() if row['obj3'] else '',
                'obj4': row['obj4'].strip() if row['obj4'] else '',
                'color1': row['color1'].strip(),
                'color2': row['color2'].strip(),
                'color3': row['color3'].strip() if row['color3'] else '',
                'color4': row['color4'].strip() if row['color4'] else '',
                'synthetic_prompt': row['synthetic_prompt'].strip()
            }
    return prompts_data


def extract_object_counts(objects_list: list[str]) -> dict[str, int]:
    """Extract object counts from the objects list."""
    return Counter(objects_list)


def normalize_object_name(obj_name: str) -> str:
    """Normalize object names for comparison (remove underscores, convert to lowercase)."""
    # Remove underscores and convert to lowercase
    normalized = obj_name.replace('_', ' ').lower().strip()
    
    # Handle special cases
    special_mappings = {
        'flat-screen tv': 'tv',
        'flat screen tv': 'tv',
        'person_sitting': 'person',
        'person sitting': 'person'
    }
    
    return special_mappings.get(normalized, normalized)


def analyze_color_prompt_structure(prompt: str, expected_color_objects: list[str]) -> bool:
    """
    Analyze if the color prompt properly describes the expected colored objects.
    Returns True if prompt seems to describe the expected colored objects, False otherwise.
    """
    prompt_lower = prompt.lower()
    
    # Check if all expected colored objects appear in the prompt
    missing_objects = []
    for colored_obj in expected_color_objects:
        if colored_obj and colored_obj.lower() not in prompt_lower:
            missing_objects.append(colored_obj)
    
    return len(missing_objects) == 0


def analyze_color_relationships(prompt: str, obj1: str, color1: str, obj2: str, color2: str, obj3: str = '', color3: str = '', obj4: str = '', color4: str = '') -> bool:
    """
    Analyze if the color prompt properly describes the expected color-object relationships.
    Returns True if the color keywords are present in the prompt.
    """
    prompt_lower = prompt.lower()
    
    # Check if all expected colors are present
    expected_colors = [color1.lower(), color2.lower()]
    if color3:
        expected_colors.append(color3.lower())
    if color4:
        expected_colors.append(color4.lower())
    
    return all(color in prompt_lower for color in expected_colors if color)


def perform_color_sanity_checks(prompt: str, box_data: list, prompts_data: dict[str, dict]) -> dict[str, list[str]]:
    """Perform sanity checks and return categorized warning messages."""
    warnings = {
        'prompt_generation': [],  # Issues with synthetic prompt generation
        'box_generation': [],     # Issues with bounding box generation
        'general': []             # General issues (bbox mismatch)
    }
    
    # We assume the prompt exists in CSV since we filter before calling this function
    csv_data = prompts_data[prompt]
    objects_list, bbox_list = box_data
    
    # Check 1: Number of objects should match number of bounding boxes within pickle file entry
    if len(objects_list) != len(bbox_list):
        warnings['general'].append(f"Objects/bboxes count mismatch: {len(objects_list)} objects but {len(bbox_list)} bounding boxes")
    
    # Check 2: Expected colored objects from CSV should be present in pickle file objects
    # Extract base objects and colors from the pickle file objects (e.g., "red banana" -> "banana")
    pickle_objects = []
    for obj in objects_list:
        obj_normalized = normalize_object_name(obj)
        pickle_objects.append(obj_normalized)
    
    # Build expected colored objects
    expected_colored_objects = []
    if csv_data['obj1'] and csv_data['color1']:
        expected_colored_objects.append(f"{csv_data['color1'].lower()} {csv_data['obj1'].lower()}")
    if csv_data['obj2'] and csv_data['color2']:
        expected_colored_objects.append(f"{csv_data['color2'].lower()} {csv_data['obj2'].lower()}")
    if csv_data['obj3'] and csv_data['color3']:
        expected_colored_objects.append(f"{csv_data['color3'].lower()} {csv_data['obj3'].lower()}")
    if csv_data['obj4'] and csv_data['color4']:
        expected_colored_objects.append(f"{csv_data['color4'].lower()} {csv_data['obj4'].lower()}")
    
    # Count expected vs actual colored objects
    expected_colored_counts = Counter(expected_colored_objects)
    pickle_colored_objects = [normalize_object_name(obj) for obj in objects_list]
    actual_colored_counts = Counter(pickle_colored_objects)
    
    for expected_colored_obj, expected_count in expected_colored_counts.items():
        actual_count = 0
        # Check for exact match or partial matches in pickle file objects
        for actual_colored_obj, count in actual_colored_counts.items():
            if expected_colored_obj in actual_colored_obj or actual_colored_obj in expected_colored_obj:
                actual_count += count
        
        if actual_count < expected_count:
            # Analyze if this is a prompt generation issue or box generation issue
            is_prompt_properly_described = analyze_color_prompt_structure(prompt, expected_colored_objects)
            
            if is_prompt_properly_described:
                # Prompt describes the expected colored objects correctly, but boxes don't match
                warnings['box_generation'].append(
                    f"[BOX GEN] Expected colored object '{expected_colored_obj}' count {expected_count}, but found only {actual_count} in pickle file"
                )
            else:
                # Prompt doesn't properly describe the expected colored objects
                warnings['prompt_generation'].append(
                    f"[PROMPT GEN] Expected colored object '{expected_colored_obj}' count {expected_count}, but color prompt doesn't properly describe this colored object (found {actual_count} in boxes)"
                )
    
    # Check 3: Color relationship validation
    if csv_data['obj1'] and csv_data['color1'] and csv_data['obj2'] and csv_data['color2']:
        is_relationship_correct = analyze_color_relationships(
            prompt, csv_data['obj1'], csv_data['color1'], csv_data['obj2'], csv_data['color2'],
            csv_data['obj3'], csv_data['color3'], csv_data['obj4'], csv_data['color4']
        )
        if not is_relationship_correct:
            warnings['prompt_generation'].append(
                "[PROMPT GEN] Color relationships not properly described in prompt"
            )
    
    return warnings


def merge_color_pickle_files_to_jsonl(pickle_files: list[str], csv_file: str, output_file: str):
    """Merge color pickle files into a single JSONL file with sanity checks."""
    print("Loading color prompts CSV...")
    prompts_data = load_color_prompts(csv_file)
    print(f"Loaded {len(prompts_data)} prompts from CSV")
    
    merged_data = []
    total_warnings = {
        'prompt_generation': 0,
        'box_generation': 0,
        'general': 0
    }
    
    for pickle_file in pickle_files:
        print(f"\nProcessing {pickle_file}...")
        box_data = load_box(pickle_file)
        print(f"Loaded {len(box_data)} entries from {pickle_file}")
        
        file_warnings = {
            'prompt_generation': 0,
            'box_generation': 0,
            'general': 0
        }
        
        skipped_entries = 0
        for prompt, data in box_data.items():
            # Skip entries that are not in the CSV prompts data (likely from different datasets)
            if prompt not in prompts_data:
                skipped_entries += 1
                print(f"🚫 SKIPPING entry not in CSV: '{prompt[:50]}...'")
                continue
                
            warnings = perform_color_sanity_checks(prompt, data, prompts_data)
            
            # Count and display warnings by category
            has_warnings = any(warnings.values())
            if has_warnings:
                print(f"⚠️  WARNING for prompt: '{prompt[:50]}...'")
                
                for category, warning_list in warnings.items():
                    if warning_list:
                        file_warnings[category] += len(warning_list)
                        total_warnings[category] += len(warning_list)

            
            # Create JSONL entry
            objects_list, bbox_list = data
            # Normalize bounding boxes from [0, 512] to [0, 1]
            normalized_bbox_list = []
            for bbox in bbox_list:
                normalized_bbox = [coord / 512.0 for coord in bbox]
                normalized_bbox_list.append(normalized_bbox)
            
            entry = {
                'prompt': prompt,
                'phrases': objects_list,
                'bounding_boxes': normalized_bbox_list,
                'num_objects': len(objects_list),
                'num_bboxes': len(bbox_list)
            }
            
            # Add CSV data (we know it exists because we checked above)
            csv_data = prompts_data[prompt]
            entry.update({
                'synthetic_prompt': csv_data['synthetic_prompt'],
                'expected_obj1': csv_data['obj1'],
                'expected_obj2': csv_data['obj2'],
                'expected_obj3': csv_data['obj3'],
                'expected_obj4': csv_data['obj4'],
                'color1': csv_data['color1'],
                'color2': csv_data['color2'],
                'color3': csv_data['color3'],
                'color4': csv_data['color4']
            })
            
            merged_data.append(entry)
        
        total_file_warnings = sum(file_warnings.values())
        print(f"Processed {pickle_file}: {total_file_warnings} warnings found, {skipped_entries} entries skipped")
        print(f"  - Prompt generation issues: {file_warnings['prompt_generation']}")
        print(f"  - Box generation issues: {file_warnings['box_generation']}")  
        print(f"  - General issues: {file_warnings['general']}")
        print(f"  - Skipped entries (not in CSV): {skipped_entries}")
    
    # Write to JSONL file
    print(f"\nWriting {len(merged_data)} entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in merged_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    total_all_warnings = sum(total_warnings.values())
    print("\n✅ Merge completed!")
    print(f"📊 Total entries: {len(merged_data)}")
    print(f"⚠️  Total warnings: {total_all_warnings}")
    if total_all_warnings > 0:
        print(f"   - Prompt generation issues: {total_warnings['prompt_generation']} ({total_warnings['prompt_generation']/total_all_warnings*100:.1f}%)")
        print(f"   - Box generation issues: {total_warnings['box_generation']} ({total_warnings['box_generation']/total_all_warnings*100:.1f}%)")
        print(f"   - General issues: {total_warnings['general']} ({total_warnings['general']/total_all_warnings*100:.1f}%)")
    print(f"💾 Output file: {output_file}")
    
    if total_all_warnings > 0:
        print("\n📊 WARNING ANALYSIS:")
        print(f"   🔍 PROMPT GENERATION issues ({total_warnings['prompt_generation']}): Color prompts don't properly describe expected objects or color relationships")
        print(f"   📦 BOX GENERATION issues ({total_warnings['box_generation']}): Prompts are correct but bounding box data doesn't match")
        print(f"   ⚙️  GENERAL issues ({total_warnings['general']}): Missing prompts or object/bbox count mismatches")
        print("\n⚠️  Please review the warnings above to ensure data quality.")


if __name__ == "__main__":
    # Define file paths
    pickle_files = [
        "gpt_generated_box/color.p"
    ]
    
    csv_file = "hrs_prompts/colors_composition_prompts.csv"
    output_file = "hrs_dataset/color.jsonl"
    
    merge_color_pickle_files_to_jsonl(pickle_files, csv_file, output_file)