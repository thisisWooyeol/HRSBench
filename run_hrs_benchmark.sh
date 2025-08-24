#!/bin/bash

METHOD=$1
IMAGE_BASE_DIR=$2
GENERATION_SEED=${3:-42}

echo "Running HRS benchmark with method: $METHOD, seed: $GENERATION_SEED"

# Find subdirectories for each task with the specified seed
FOUND_TASK_DIRS=()
TASKS=("counting" "spatial" "size" "color")

for TASK in "${TASKS[@]}"; do
    TASK_PATTERN="${TASK}_seed${GENERATION_SEED}"
    TASK_DIR="${IMAGE_BASE_DIR}/${TASK_PATTERN}"
    
    if [ -d "$TASK_DIR" ]; then
        echo "Found task directory: $TASK_DIR"
        FOUND_TASK_DIRS+=("$TASK_DIR")
    else
        echo "Task directory not found: $TASK_DIR"
    fi
done

echo "Found task directories with seed $GENERATION_SEED:"
for TASK_DIR in "${FOUND_TASK_DIRS[@]}"; do
    echo " - $TASK_DIR"
done

if [ ${#FOUND_TASK_DIRS[@]} -eq 0 ]; then
    echo "No task directories found with seed $GENERATION_SEED. Exiting."
    exit 1
fi

# Create output directory for the method
OUTPUT_DIR="./output/${METHOD}_seed${GENERATION_SEED}"
mkdir -p "$OUTPUT_DIR"

# Run benchmarking for each task
for TASK_DIR in "${FOUND_TASK_DIRS[@]}"; do
    # Extract task type from directory name
    TASK_NAME=$(basename "$TASK_DIR")
    TASK_TYPE=$(echo "$TASK_NAME" | sed "s/_seed${GENERATION_SEED}//")
    
    echo ""
    echo "=== Processing $TASK_TYPE task ==="
    
    case $TASK_TYPE in
        "counting"|"spatial"|"size")
            echo "Running UniDet detection for $TASK_TYPE..."
            
            # Run UniDet detection
            PKL_PATH="${OUTPUT_DIR}/${TASK_TYPE}.pkl"
            python src/detection/UniDet-master/demo.py \
                --input "${TASK_DIR}/*" \
                --output_base_dir "$OUTPUT_DIR" \
                --task "$TASK_TYPE" \
                --pkl_pth "$PKL_PATH" \
                --opts MODEL.WEIGHTS "Partitioned_COI_RS101_2x.pth"
            
            if [ $? -eq 0 ]; then
                echo "UniDet detection completed successfully for $TASK_TYPE"
                
                # Run accuracy calculation based on task type
                case $TASK_TYPE in
                    "counting")
                        echo "Calculating counting accuracy..."
                        python src/counting/calc_counting_acc.py --in_pkl_path "$PKL_PATH"
                        ;;
                    "spatial")
                        echo "Calculating spatial composition accuracy..."
                        python src/compositions/calc_spatial_relation_acc.py --in_pkl_path "$PKL_PATH"
                        ;;
                    "size")
                        echo "Calculating size composition accuracy..."
                        python src/compositions/calc_size_comp_acc.py --in_pkl_path "$PKL_PATH"
                        ;;
                esac
            else
                echo "Error: UniDet detection failed for $TASK_TYPE"
            fi
            ;;
            
        "color")
            echo "Running MaskDINO segmentation for color task..."
            
            # Run MaskDINO segmentation
            python src/colors/MaskDINO/demo/demo.py \
                --input "${TASK_DIR}/*" \
                --output_base_dir "$OUTPUT_DIR" \
                --opts MODEL.WEIGHTS "maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth"
            
            if [ $? -eq 0 ]; then
                echo "MaskDINO segmentation completed successfully for color task"
                
                # Run color classification
                echo "Running color classification..."
                python src/colors/hue_based_color_classifier.py \
                    --input_image_dir "$TASK_DIR" \
                    --input_mask_dir "${OUTPUT_DIR}/color_detected_images"
            else
                echo "Error: MaskDINO segmentation failed for color task"
            fi
            ;;
        *)
            echo "Unknown task type: $TASK_TYPE"
            ;;
    esac
done

echo ""
echo "=== HRS Benchmark completed for method: $METHOD ==="
echo "Results saved in: $OUTPUT_DIR"