import pickle
import csv
import json
from collections import Counter


def load_box(pickle_file: str) -> dict[str, list]:
    """Load pickle file and return the data dictionary."""
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data


def load_counting_prompts(csv_file: str) -> dict[str, dict]:
    """Load the counting prompts CSV and return a dictionary with synthetic_prompt as key."""
    prompts_data = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            synthetic_prompt = row['synthetic_prompt'].strip()
            prompts_data[synthetic_prompt] = {
                'n1': int(row['n1']) if row['n1'] else 0,
                'obj1': row['obj1'].strip(),
                'n2': int(row['n2']) if row['n2'] else 0,
                'obj2': row['obj2'].strip(),
                'vanilla_prompt': row['vanilla_prompt'].strip()
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


def analyze_prompt_vs_expected(prompt: str, expected_obj: str, expected_count: int) -> bool:
    """
    Analyze if the synthetic prompt properly describes the expected objects.
    Returns True if prompt seems to describe the expected count, False otherwise.
    """
    prompt_lower = prompt.lower()
    obj_lower = expected_obj.lower()
    
    # Check if the expected object appears in the prompt
    if obj_lower not in prompt_lower:
        return False
    
    # Look for number patterns in the prompt that match expected count
    import re
    
    # Number words mapping
    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    
    # Find all numbers (digits and words) in the prompt
    digit_matches = re.findall(r'\b\d+\b', prompt_lower)
    word_matches = re.findall(r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b', prompt_lower)
    
    all_numbers = []
    all_numbers.extend([int(x) for x in digit_matches])
    all_numbers.extend([number_words[x] for x in word_matches])
    
    # If expected count appears in the prompt, assume it's properly described
    if expected_count in all_numbers:
        return True
    
    # Special case: if expected count is 1, and no numbers are mentioned, it might be implicit
    if expected_count == 1 and not all_numbers:
        # Check for singular/plural forms
        if obj_lower.endswith('s') or 'are' in prompt_lower or 'two' in prompt_lower or 'multiple' in prompt_lower:
            return False  # Prompt suggests plural but expected is 1
        return True
    
    return False


def perform_sanity_checks(prompt: str, box_data: list, prompts_data: dict[str, dict]) -> dict[str, list[str]]:
    """Perform sanity checks and return categorized warning messages."""
    warnings = {
        'prompt_generation': [],  # Issues with synthetic prompt generation
        'box_generation': [],     # Issues with bounding box generation
        'general': []             # General issues (prompt not found, bbox mismatch)
    }
    
    # Check 1: Prompt should exist in CSV
    if prompt not in prompts_data:
        warnings['general'].append(f"Prompt not found in CSV: '{prompt}'")
        return warnings
    
    csv_data = prompts_data[prompt]
    objects_list, bbox_list = box_data
    
    # Check 2: Number of objects should match number of bounding boxes within pickle file entry
    if len(objects_list) != len(bbox_list):
        warnings['general'].append(f"Objects/bboxes count mismatch: {len(objects_list)} objects but {len(bbox_list)} bounding boxes")
    
    # Check 3: Expected objects from CSV should be present in pickle file objects
    # Count objects in the pickle file
    object_counts = extract_object_counts([normalize_object_name(obj) for obj in objects_list])
    
    # Check expected objects are present with correct counts
    expected_objects = []
    if csv_data['n1'] > 0 and csv_data['obj1']:
        obj1_normalized = normalize_object_name(csv_data['obj1'])
        expected_objects.append((obj1_normalized, csv_data['n1']))
    
    if csv_data['n2'] > 0 and csv_data['obj2']:
        obj2_normalized = normalize_object_name(csv_data['obj2'])
        expected_objects.append((obj2_normalized, csv_data['n2']))
    
    for expected_obj, expected_count in expected_objects:
        actual_count = 0
        # Check for exact match or partial matches in pickle file objects
        for actual_obj, count in object_counts.items():
            if expected_obj in actual_obj or actual_obj in expected_obj:
                actual_count += count
        
        if actual_count < expected_count:
            # Analyze if this is a prompt generation issue or box generation issue
            is_prompt_properly_described = analyze_prompt_vs_expected(prompt, expected_obj, expected_count)
            
            if is_prompt_properly_described:
                # Prompt describes the expected count correctly, but boxes don't match
                warnings['box_generation'].append(
                    f"[BOX GEN] Expected object '{expected_obj}' count {expected_count}, but found only {actual_count} in pickle file"
                )
            else:
                # Prompt doesn't properly describe the expected count
                warnings['prompt_generation'].append(
                    f"[PROMPT GEN] Expected object '{expected_obj}' count {expected_count}, but synthetic prompt doesn't properly describe this count (found {actual_count} in boxes)"
                )
    
    return warnings


def merge_pickle_files_to_jsonl(pickle_files: list[str], csv_file: str, output_file: str):
    """Merge pickle files into a single JSONL file with sanity checks."""
    print("Loading counting prompts CSV...")
    prompts_data = load_counting_prompts(csv_file)
    print(f"Loaded {len(prompts_data)} prompts from CSV")
    
    merged_data = []
    total_warnings = {
        'prompt_generation': 0,
        'box_generation': 0,
        'general': 0
    }
    
    # Track processed prompts to skip duplicates
    processed_prompts = set()
    
    for pickle_file in pickle_files:
        print(f"\nProcessing {pickle_file}...")
        box_data = load_box(pickle_file)
        print(f"Loaded {len(box_data)} entries from {pickle_file}")
        
        file_warnings = {
            'prompt_generation': 0,
            'box_generation': 0,
            'general': 0
        }
        
        skipped_duplicates = 0
        for prompt, data in box_data.items():
            # Skip duplicate prompts to maintain consistency with unique CSV prompts
            if prompt in processed_prompts:
                skipped_duplicates += 1
                continue
            
            processed_prompts.add(prompt)
            warnings = perform_sanity_checks(prompt, data, prompts_data)
            
            # Count and display warnings by category
            has_warnings = any(warnings.values())
            if has_warnings:
                print(f"⚠️  WARNING for prompt: '{prompt[:50]}...'")
                
                for category, warning_list in warnings.items():
                    if warning_list:
                        file_warnings[category] += len(warning_list)
                        total_warnings[category] += len(warning_list)
                        for warning in warning_list:
                            print(f"   - {warning}")
            
            # Create JSONL entry
            objects_list, bbox_list = data
            # Normalize bounding boxes from [0, 512] to [0, 1]
            normalized_bbox_list = []
            for bbox in bbox_list:
                normalized_bbox = [coord / 512.0 for coord in bbox]
                normalized_bbox_list.append(normalized_bbox)            
            

            if prompt in prompts_data:
                csv_data = prompts_data[prompt]

                # Determine level based on n1, n2
                if csv_data['n1'] == 0 or csv_data['n2'] == 0:
                    level = 1
                elif (csv_data['n1'] > 0 and csv_data['n1'] < 4) and (csv_data['n2'] > 0 and csv_data['n2'] < 4):
                    level = 2
                elif (csv_data['n1'] >= 4 and csv_data['n2'] >= 4):
                    level = 3
                else: 
                    warnings['general'].append(f"Uncategorized level for n1={csv_data['n1']}, n2={csv_data['n2']}")
                    continue

                entry = {
                    'prompt': prompt,
                    'phrases': objects_list,
                    'bounding_boxes': normalized_bbox_list,
                    'num_objects': len(objects_list),
                    'num_bboxes': len(bbox_list),
                    'vanilla_prompt': csv_data['vanilla_prompt'],
                    'expected_n1': csv_data['n1'],
                    'expected_obj1': csv_data['obj1'],
                    'expected_n2': csv_data['n2'],
                    'expected_obj2': csv_data['obj2'],
                    'level': level
                }
            
                merged_data.append(entry)
        
        total_file_warnings = sum(file_warnings.values())
        print(f"Processed {pickle_file}: {total_file_warnings} warnings found, {skipped_duplicates} duplicates skipped")
        print(f"  - Prompt generation issues: {file_warnings['prompt_generation']}")
        print(f"  - Box generation issues: {file_warnings['box_generation']}")  
        print(f"  - General issues: {file_warnings['general']}")
        print(f"  - Skipped duplicates: {skipped_duplicates}")
    
    # Calculate total duplicates skipped across all files
    total_pickle_entries = sum(len(load_box(pf)) for pf in pickle_files)
    total_duplicates_skipped = total_pickle_entries - len(merged_data)
    
    # Write to JSONL file
    print(f"\nWriting {len(merged_data)} entries to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in merged_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    total_all_warnings = sum(total_warnings.values())
    print("\n✅ Merge completed!")
    print(f"📊 Total entries processed: {len(merged_data)}")
    print(f"🔄 Total duplicates skipped: {total_duplicates_skipped}")
    print(f"⚠️  Total warnings: {total_all_warnings}")
    if total_all_warnings > 0:
        print(f"   - Prompt generation issues: {total_warnings['prompt_generation']} ({total_warnings['prompt_generation']/total_all_warnings*100:.1f}%)")
        print(f"   - Box generation issues: {total_warnings['box_generation']} ({total_warnings['box_generation']/total_all_warnings*100:.1f}%)")
        print(f"   - General issues: {total_warnings['general']} ({total_warnings['general']/total_all_warnings*100:.1f}%)")
    print(f"💾 Output file: {output_file}")
    
    if total_all_warnings > 0:
        print("\n📊 WARNING ANALYSIS:")
        print(f"   🔍 PROMPT GENERATION issues ({total_warnings['prompt_generation']}): Synthetic prompts don't properly describe the expected object counts")
        print(f"   📦 BOX GENERATION issues ({total_warnings['box_generation']}): Prompts are correct but bounding box data doesn't match")
        print(f"   ⚙️  GENERAL issues ({total_warnings['general']}): Missing prompts or object/bbox count mismatches")
        print("\n⚠️  Please review the warnings above to ensure data quality.")


if __name__ == "__main__":
    # Define file paths
    pickle_files = [
        "legacy/gpt_generated_box/counting_0_499.p",
        "legacy/gpt_generated_box/counting_500_1499.p", 
        "legacy/gpt_generated_box/counting_1500_2499.p",
        "legacy/gpt_generated_box/counting_2500_2999.p"
    ]
    
    csv_file = "legacy/hrs_prompts/counting_prompts.csv"
    output_file = "hrs_dataset/counting.jsonl"
    
    merge_pickle_files_to_jsonl(pickle_files, csv_file, output_file)
