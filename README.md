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