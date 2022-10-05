# Unfiltered version

This removes the NSFW filter to improve performance of image generation when used in batch mode.


## xformers speedup

I merged in the speedup from this PR:
https://github.com/huggingface/diffusers/pull/532

To use it:

```
export USE_MEMORY_EFFICIENT_ATTENTION=1
pip install git+https://github.com/facebookresearch/xformers@51dd119#egg=xformers
```


## Setup

```
pip install .
```

I also had to install pytorch locally with pip: https://pytorch.org/get-started/locally/


## GPU VRAM Considerations

With the xformers speedup above, this only uses 10 GB VRAM even with batches of 12 at fp32.


## Generate Images

To generate images on a specific GPU in a multi-GPU rig, pass an argument:

```
python3 generate_images.py 2
```


## Generate Images with CLIP-guided Stable Diffusion pipeline


```
USE_MEMORY_EFFICIENT_ATTENTION=1  CUDA_VISIBLE_DEVICES=0 python3 clip_generate.py
```

