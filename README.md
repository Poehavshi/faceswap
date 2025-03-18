# FaceSwap

This is a simple online face swapping solution, which is based on ACE++

## Description

All results can be found at faceswap/data/faceswaps

Pipeline consist of several main steps:

### 1. Face aligment

It's located under faceswap/preprocess/align.py.

Main logic is to detect face and landmarks. And after align face by eyes and lips landmarks.
It utilizes RetinaFace and Landmark106 models from [insight face](https://github.com/deepinsight/insightface).
Also, there are option to use [mediapipe](https://github.com/google-ai-edge/mediapipe) as landmark model, but it produces lower quality of alignment (settings located at faceswap/preprocess/config.py)


Example of steps before and after:

Before:

![IMG_1582.jpg](faceswap/data/raw/IMG_1582.jpg)

After:

![IMG_1582.png](faceswap/data/aligned/IMG_1582.png)

Before:

![IMG_6909.jpg](faceswap/data/raw/IMG_6909.jpg)

After:

![IMG_6909.png](faceswap/data/aligned/IMG_6909.png)

### 2. Faceswap

Second step utilize ACE++ model with FluxFill and [portrait lora](https://huggingface.co/ali-vilab/ACE_Plus/tree/main/portrait) from the authors of ACE++

Comfy workflow with the same pipeline can be founded in faceswap/inpaint/workflow/Flux_ACE++.json

Input:

![IMG_1582.png](faceswap/data/aligned/IMG_1582.png)

Output:

![IMG_1582.jpg](faceswap/data/faceswaps/IMG_1582.jpg)


Input:

![IMG_0653.png](faceswap/data/aligned/IMG_0653.png)

Result:

![IMG_0653.png](faceswap/data/faceswaps/IMG_0653.png)

### Enhancment



## Installation

```bash
pip install -e .
```

## Usage

```bash
python faceswap --source <source_image> --target <target_image> --output <output_image>
```
