# HRS-Bench: Holistic, Reliable and Scalable Benchmark for Text-to-Image Models

[**"HRS-Bench: Holistic, Reliable and Scalable Benchmark for Text-to-Image Models"**](https://arxiv.org/abs/2304.05390)

[Eslam Abdelrahman](https://eslambakr.github.io/),
[Pengzhan Sun](https://pengzhansun.github.io/),
[Xiaoqian Shen](https://scholar.google.com/citations?user=uToGtIwAAAAJ&hl=en),
[Faizan Farooq Khan](https://scholar.google.com/citations?user=vwEC-jUAAAAJ&hl=en),
[Li Erran Li](https://scholar.google.com/citations?user=GkMfzy4AAAAJ&hl=en),
and [Mohamed Elhoseiny](https://scholar.google.com/citations?user=iRBUTOAAAAAJ&hl=en)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://eslambakr.github.io/hrsbench.github.io/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2304.05390)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://hrsbench.github.io/index.html)

This is the forked version of HRSBench, modified from [Attention-Refocusing](https://github.com/Attention-Refocusing/attention-refocusing) paper. User can benchmark both text-to-image and box layout-to-image generation tasks.
For box layout-to-image tasks, GPT4 generated layouts which are provided by `Attention-Refocusing` repository are used.

---
<br>

# About HRS dataset

The core functionality of HRS-Bench are included in `src` directory, majorly copied from [Attention-Refocusing](https://github.com/Attention-Refocusing/attention-refocusing).

With the HRS prompts and GPT-4 generated box layouts from Attention-Refocusing, I pre-processed the complete dataset into `hrs_dataset` in `jsonl` format. 

The dataset includes four main categories: spatial relationship, color, size, and counting. Even though HRS prompts for each category counting/spatial/size/color are 3, 000/1,002/501/501, there are some duplicated prompts. If we count only unique prompts, the numbers are 2,990/898/424/484.

And there are also mysteriously missing prompts in `*.p` pickle files, last lines for all datasets. Therefore, 2,990/896/423/483 unique prompts can be evaluated.


# How to evaluate

Setup virtual environment first:
```bash
uv sync
source .venv/bin/activate
```

## Step-1: Generate Images with HRS prompts (and optionally with box layouts)

The HRS Bench dataset is structured to support benchmarking for both text-to-image (T2I) and layout-to-image (L2I) tasks. The dataset includes **prompts** and corresponding **box layouts** for each image. Users can choose to generate images using either the prompts alone or the prompts in conjunction with the box layouts.

All dataset specifications, including prompt formats and layout details, are provided in `hrs_dataset/` in `jsonl` format. Here, we will show core components that users need to be aware of when working with the dataset.

- `prompt`: The text prompt used for image generation.
- `phrases`: Simple descriptions of its corresponding box layout.
- `bounding_boxes`: The box layout information, 0-1 scale tuple4 (x_min, y_min, x_max, y_max)

### ℹ️ Additional information for ISAC

- about tags
  - `expected_obj1` to `expected_obj4` (maximum) provides prompt including object tags for each image. These tags are crucial for tag based *Attention Modulation*.

- about instance count 
  - For `spatial`, `size` and `color` tasks, only one instance is presented per each object category. This means, `n=1` is the default setting for these tasks.
  - For `counting` tasks, multiple instances may be presented for maximum two object categories. You may refer `expected_n1` and `expected_n2` to retrieve instance counts. (GT values) 

### Naming rule for HRS dataset

Each generated image for a specific task should be saved in a separate folder, sharing the same parent directory. And generated images should follow the naming convention: `<prompt_idx>_<level>_<prompt>.[png|jpg]` For example:

```
/path/to/IMAGE_ROOT/
├── color_seed42/
├── counting_seed42/
├── size_seed42/
└── spatial_seed42/
```

## Step-2: Evaluate Generated Images

`run_hrs_benchmark.sh` provides an easy way to evaluate the generated images against the HRS dataset. This script will automatically run the evaluation process, saving the results in a specified output directory.

More details about the intermediate process can be found in [`README.md`](src/README.md) in `src` directory.

Here is the way to run the benchmark script:

```bash
bash run_hrs_benchmark.sh <METHOD_NAME> <IMAGE_ROOT> <GENERATION_SEED>
```

where
- `<METHOD_NAME>`: The name of the method to use for evaluation. This will be used for creating output directory name (e.g., `SD1.5`).
- `<IMAGE_ROOT>`: The root directory containing the generated images for evaluation.
- `<GENERATION_SEED>`: The seed used for image generation, which helps in reproducing results.