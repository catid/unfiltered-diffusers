# make sure you're logged in with `huggingface-cli login`
from torch import autocast
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import time
import random


prompt = "rock climbing cat"

seed = random.randint(0, 2000000000)
steps = 60
h = 512
w = 512
batch = 6



lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    scheduler=lms,
    use_auth_token=True
)
pipe = pipe.to("cuda")


for j in range(1, 100):

    seed_n = seed + j

    start = time.time()
    generator = torch.Generator("cuda").manual_seed(seed_n)
    with autocast("cuda"):
        images = pipe([prompt] * batch, num_inference_steps=steps, height=h, width=w, generator=generator)["sample"]
   
        for i, img in enumerate(images):
            img.save("{}_{}_{}.png".format(prompt, seed_n, i)) 

    end = time.time()
    print("Time to complete {} images: {} seconds".format(batch, end - start))
