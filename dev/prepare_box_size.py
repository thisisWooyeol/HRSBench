import pickle
import csv
import json
from collections import Counter


def load_box(pickle_file: str) -> dict[str, list]:
    """Load pickle file and return the data dictionary."""
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    return data


def load_size_prompts(csv_file: str) -> dict[str, dict]:
    """Load the size prompts CSV and return a dictionary with meta_prompt as key."""
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
                'rel1': row['rel1'].strip(),
                'rel2': row['rel2'].strip() if row['rel2'] else '',
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


def analyze_size_prompt_structure(prompt: str, expected_objects: list[str]) -> bool:
    """
    Analyze if the size prompt properly describes the expected objects.
    Returns True if prompt seems to describe the expected objects, False otherwise.
    """
    prompt_lower = prompt.lower()
    
    # Check if all expected objects appear in the prompt
    missing_objects = []
    for obj in expected_objects:
        if obj and obj.lower() not in prompt_lower:
            missing_objects.append(obj)
    
    return len(missing_objects) == 0


def analyze_size_relationships(prompt: str, obj1: str, obj2: str, rel1: str) -> bool:
    """
    Analyze if the size prompt properly describes the expected size relationship.
    Returns True if the relationship keyword is present in the prompt.
    """
    prompt_lower = prompt.lower()
    rel1_lower = rel1.lower()
    
    # Simply check if the relationship keyword is in the prompt
    return rel1_lower in prompt_lower


def perform_size_sanity_checks(prompt: str, box_data: list, prompts_data: dict[str, dict]) -> dict[str, list[str]]:
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
    
    # Check 2: Expected objects from CSV should be present in pickle file objects
    # Count objects in the pickle file
    object_counts = extract_object_counts([normalize_object_name(obj) for obj in objects_list])
    
    # Collect all expected objects
    expected_objects = []
    if csv_data['obj1']:
        expected_objects.append(normalize_object_name(csv_data['obj1']))
    if csv_data['obj2']:
        expected_objects.append(normalize_object_name(csv_data['obj2']))
    if csv_data['obj3']:
        expected_objects.append(normalize_object_name(csv_data['obj3']))
    if csv_data['obj4']:
        expected_objects.append(normalize_object_name(csv_data['obj4']))
    
    # Count expected vs actual objects
    expected_object_counts = Counter(expected_objects)
    
    for expected_obj, expected_count in expected_object_counts.items():
        actual_count = 0
        # Check for exact match or partial matches in pickle file objects
        for actual_obj, count in object_counts.items():
            if expected_obj in actual_obj or actual_obj in expected_obj:
                actual_count += count
        
        if actual_count < expected_count:
            # Analyze if this is a prompt generation issue or box generation issue
            is_prompt_properly_described = analyze_size_prompt_structure(prompt, expected_objects)
            
            if is_prompt_properly_described:
                # Prompt describes the expected objects correctly, but boxes don't match
                warnings['box_generation'].append(
                    f"[BOX GEN] Expected object '{expected_obj}' count {expected_count}, but found only {actual_count} in pickle file"
                )
            else:
                # Prompt doesn't properly describe the expected objects
                warnings['prompt_generation'].append(
                    f"[PROMPT GEN] Expected object '{expected_obj}' count {expected_count}, but size prompt doesn't properly describe this object (found {actual_count} in boxes)"
                )
    
    # Check 3: Size relationship validation
    if csv_data['obj1'] and csv_data['obj2'] and csv_data['rel1']:
        is_relationship_correct = analyze_size_relationships(
            prompt, csv_data['obj1'], csv_data['obj2'], csv_data['rel1']
        )
        if not is_relationship_correct:
            warnings['prompt_generation'].append(
                f"[PROMPT GEN] Size relationship '{csv_data['obj1']} {csv_data['rel1']} {csv_data['obj2']}' not properly described in prompt"
            )
    
    return warnings


def merge_size_pickle_files_to_jsonl(pickle_files: list[str], csv_file: str, output_file: str):
    """Merge size pickle files into a single JSONL file with sanity checks."""
    print("Loading size prompts CSV...")
    prompts_data = load_size_prompts(csv_file)
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
                
            warnings = perform_size_sanity_checks(prompt, data, prompts_data)
            
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
                'relation1': csv_data['rel1'],
                'relation2': csv_data['rel2']
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
        print(f"   🔍 PROMPT GENERATION issues ({total_warnings['prompt_generation']}): Size prompts don't properly describe expected objects or size relationships")
        print(f"   📦 BOX GENERATION issues ({total_warnings['box_generation']}): Prompts are correct but bounding box data doesn't match")
        print(f"   ⚙️  GENERAL issues ({total_warnings['general']}): Missing prompts or object/bbox count mismatches")
        print("\n⚠️  Please review the warnings above to ensure data quality.")


if __name__ == "__main__":
    # Define file paths
    pickle_files = [
        "gpt_generated_box/size.p"
    ]
    
    csv_file = "hrs_prompts/size_compositions_prompts.csv"
    output_file = "hrs_dataset/size.jsonl"
    
    merge_size_pickle_files_to_jsonl(pickle_files, csv_file, output_file)
