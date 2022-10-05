# Unfiltered version

This removes the NSFW filter to improve performance of image generation when used in batch mode.


## Setup

```
pip install .
```

I also had to install pytorch locally with pip: https://pytorch.org/get-started/locally/


## GPU VRAM Considerations

My test scripts are designed for RTX 3090 which has 24GB VRAM and can batch 6 images at fp32 precision.

You can switch to fp16 and/or reduce the batch size for other GPUs.


## Generate Images with K-LMS Stable Diffusion pipeline

To generate images on a specific GPU in a multi-GPU rig:

```
CUDA_VISIBLE_DEVICES=0 python3 generate_images.py
```


## Generate Images with CLIP-guided Stable Diffusion pipeline


```
CUDA_VISIBLE_DEVICES=0 python3 clip_generate.py
```

## xformers speedup

I merged in the speedup from this PR:
https://github.com/huggingface/diffusers/pull/532

To use it:

```
export USE_MEMORY_EFFICIENT_ATTENTION=1
pip install git+https://github.com/facebookresearch/xformers@51dd119#egg=xformers
```

Then reinstall this version of diffusers, and it should activate the feature.

