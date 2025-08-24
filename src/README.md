First, install [detectron2](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).
Then, download the [UniDet](https://github.com/xingyizhou/UniDet)'s
[weights](https://drive.google.com/file/d/110JSpmfNU__7T3IMSJwv0QSfLLo_AqtZ)
and [configurations](https://github.com/xingyizhou/UniDet/blob/master/configs/Partitioned_COI_RS101_2x.yaml).

## Counting, Spatail, Size Composition

Run the
[inference code](detection/UniDet-master/demo.py)
to generate the bounding boxes and save them as follows:
```bash
python src/detection/UniDet-master/demo.py \
    --input "/mnt/nas5/AIBL-Research/shjo/250809_ISAC/output/GLIGEN/counting_seed42/*" \
    --output_base_dir "./output/GLIGEN" \
    --task "counting" \
    --pkl_pth "./output/GLIGEN/counting.pkl" \
    --opts MODEL.WEIGHTS "Partitioned_COI_RS101_2x.pth"

python src/detection/UniDet-master/demo.py \
    --input "/mnt/nas5/AIBL-Research/shjo/250809_ISAC/output/GLIGEN/spatial_seed42/*" \
    --output_base_dir "./output/GLIGEN" \
    --task "spatial" \
    --pkl_pth "./output/GLIGEN/spatial.pkl" \
    --opts MODEL.WEIGHTS "Partitioned_COI_RS101_2x.pth"

python src/detection/UniDet-master/demo.py \
    --input "/mnt/nas5/AIBL-Research/shjo/250809_ISAC/output/GLIGEN/size_seed42/*" \
    --output_base_dir "./output/GLIGEN" \
    --task "size" \
    --pkl_pth "./output/GLIGEN/size.pkl" \
    --opts MODEL.WEIGHTS "Partitioned_COI_RS101_2x.pth"
```
Where:
- `--input`: Folder of images that need to be evaluated
- `--output_base_dir`: Path to the folder to save visually detected images
- `--task`: The task to be performed (counting, spatial, size)
- `--pkl_pth`: The path to the output .pkl file to save detected information
- `--opts`: Path to the weights downloaded above


### Counting 
Run the 
[calc_counting_acc.py](counting/calc_counting_acc.py)
to calculate the counting accuracy, as follows:
```bash
python src/counting/calc_counting_acc.py --in_pkl_path "./output/GLIGEN/counting.pkl"
```

### Spatial composition
Run the 
[calc_spatial_relation_acc.py](compositions/calc_spatial_relation_acc.py)
to calculate the spatial composition accuracy, as follows:
```bash
python src/compositions/calc_spatial_relation_acc.py --in_pkl_path "./output/GLIGEN/spatial.pkl"
```

### Size
Run the 
[calc_size_comp_acc.py](compositions/calc_size_comp_acc.py)
to calculate the size composition accuracy, as follows:
```bash
python src/compositions/calc_size_comp_acc.py --in_pkl_path "./output/GLIGEN/size.pkl"
```

Where `--in_pkl_path` is the output file from running UniDet as shown in the above steps.


## Color Composition:
We adopt [MaskDINO](https://arxiv.org/pdf/2206.02777.pdf) [CVPR 2023] for the instance segmentation.
Download the [model weight](https://github.com/IDEA-Research/detrex-storage/releases/download/maskdino-v0.1.0/maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth) and the corresponding [config](colors/MaskDINO/configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml).
Before run, please compile `MultiScaleDeformableAttention` CUDA op with the following commands:
```bash
cd src/colors/MaskDINO/maskdino/modeling/pixel_decoder/ops
sh make.sh
```

Now run the 
[inference code](colors/MaskDINO/demo/demo.py)
to predict the masks for each instance and save them as follows:
```bash
python src/colors/MaskDINO/demo/demo.py \
    --input "/mnt/nas5/AIBL-Research/shjo/250809_ISAC/output/GLIGEN/color_seed42/*" \
    --output_base_dir  "./output/GLIGEN" \
    --opts MODEL.WEIGHTS "maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth"
```
Where:

- `--input`: Directory: path to folder images need to evaluate
- `--output_base_dir`: Directory: path to output of segmented masks
- `--opts`: path to the model weight downloaded above


Then, run the 
[hue_based_color_classifier.py](colors/hue_based_color_classifier.py)
to calculate the color composition accuracy, as follows:
```bash
python src/colors/hue_based_color_classifier.py \
    --input_image_dir "/mnt/nas5/AIBL-Research/shjo/250809_ISAC/output/SD1.5/color_seed42" \
    --input_mask_dir "/mnt/nas5/Public Dataset/HRSBench/output/SD1.5/color_detected_images"
```
Where:
- `--input_image_dir`: Directory: path to folder of originally generated images
- `--input_mask_dir`: Directory: path to folder of generated segmentation masks