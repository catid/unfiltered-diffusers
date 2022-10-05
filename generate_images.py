# make sure you're logged in with `huggingface-cli login`
from torch import autocast
import torch
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import time
import random
import os


seed = random.randint(0, 2000000000)
steps = 100
h = 512
w = 512
batch = 1
n=50 # Number of image pairs to generate

model = "CompVis/stable-diffusion-v1-4"
#model = "hakurei/waifu-diffusion"



dir_name = "conch_flower4"

text_prompt = "conch shell near colorful flowers, white frame"

text_prompt = text_prompt + ", art by rossdraws" # Bright, illustrative, digital airbrush
#text_prompt = text_prompt + ", art by victo ngai" # Vector, clean, thin lines
#text_prompt = text_prompt + ", art by thomas kinkade" # Painting, scenery
#text_prompt = text_prompt + ", art by ilya kuvshinov" # High contrast character poses, provocative female
#text_prompt = text_prompt + ", art by peter mohrbacher" # Vector, scenery, fantasy
#text_prompt = text_prompt + ", art by craig mullins" # Painting, scenery, sci-fi
#text_prompt = text_prompt + ", art by stanley artgerm lau" # Comic book, super hero full body, provocative female
#text_prompt = text_prompt + ", art by wlop" # Painting, character studies, fantasy, high class
#text_prompt = text_prompt + ", art by james jean" # Abstract, bright pop, colorful
#text_prompt = text_prompt + ", art by andrei riabovitchev" # Abstract, sci-fi worlds and sitting portraits with lots of clothing detail
#text_prompt = text_prompt + ", art by marc simonetti" # Intricate sci-fi worlds
#text_prompt = text_prompt + ", art by henry vargas" # Low poly colorful 3D CG game art
#text_prompt = text_prompt + ", art by studio ghibli, art by Hayao Miyazaki" # Ghibli
#text_prompt = text_prompt + ", art by greg rutkowski" # Fantasy
#text_prompt = text_prompt + ", art by alphonse mucha" # Old timey card/poster
#text_prompt = text_prompt + ", art by disney" # Disney
#text_prompt = text_prompt + ", art by pixar" # Pixar
#text_prompt = text_prompt + ", art by Sophie Anderson" # 
#text_prompt = text_prompt + ", art by Lise Deharme" # 
#text_prompt = text_prompt + ", art by Peter Kemp" # 
#text_prompt = text_prompt + ", art by Storm Thorgerson" # 

#text_prompt = text_prompt + ", artstation" # Illustration artwork
#text_prompt = text_prompt + ", digital art, illustration"

#text_prompt = text_prompt + ", cgsociety" # CG artwork
#text_prompt = text_prompt + ", unreal engine 5, blender, octane, ray tracing"

#text_prompt = text_prompt + ", painting"
#text_prompt = text_prompt + ", smooth"
#text_prompt = text_prompt + ", concept art"
#text_prompt = text_prompt + ", photograph"
#text_prompt = text_prompt + ", sharp focus, high definition, 4k resolution, 8k resolution, highly detailed, intricate"

#text_prompt = text_prompt + ", dramatic studio lighting"
#text_prompt = text_prompt + ", masterpiece, post processing"

#text_prompt = text_prompt + ", headshot, portrait"
#text_prompt = text_prompt + ", symmetrical composition"
#text_prompt = text_prompt + ", tilt-shift photography"
#text_prompt = text_prompt + ", black and white"

#text_prompt = "A photo-real delicate iridescent ceramic porcelain sculpture of an ornate detailed symmetrical wolf head in front if an intricate background by Victo Ngai and takato yamamoto, AJ Fosik, symmetrical composition, backlit lighting, subsurface scattering, translucent, thin porcelain, opaline eyes"
#text_prompt = text_prompt + ", octane renderer, colorful, physically based rendering, trending on cgsociety"

#text_prompt = "a cat in an iron man suit healthy animation shot cute studio ghibli pixar and disney animation rendered in unreal engine crisp anime key art by greg rutkowski dramatic lighting 4k resolution"
#text_prompt = "symmetrical portrait of emperor palpatine as a cat, healthy animation shot cute studio ghibli pixar and disney animation rendered in unreal engine crisp anime key art by greg rutkowski dramatic lighting 4k resolution"
#text_prompt = text_prompt + ", green background"

# hollow knight, vinyl toy figurine octane render by henry vargas

#text_prompt = "hollow knight, vinyl toy figurine octane render by henry vargas"

# rocket_as_a_girl_made_by_stanley_artgerm_lau_wlop_rossdraws_james_jean_andrei_riabovitchev_marc_simonetti_yoshitaka_amano_art
# portrait_of_megan_fox_with_glasses_in_a_black_business_suit_with_white_shirt_and_black_cravat_intricate_headshot_highly_detailed_digita_


#text_prompt = "robotic creature kitchen on a spaceship sci-fi by Justin Gerard highly detailed movie shot"













lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

pipe = StableDiffusionPipeline.from_pretrained(
    model,
    scheduler=lms,
    use_auth_token=True
)
pipe = pipe.to("cuda")








os.makedirs(dir_name, exist_ok=True)


for j in range(0, n-1):

    seed_n = seed + j

    start = time.time()
    generator = torch.Generator("cuda").manual_seed(seed_n)
    with autocast("cuda"):
        images = pipe([text_prompt] * batch, num_inference_steps=steps, height=h, width=w, generator=generator)["sample"]
   
        for i, img in enumerate(images):
            img.save("{}/{}_{}.png".format(dir_name, seed_n, i))

    end = time.time()
    print("Time to complete {} images: {} seconds".format(batch, end - start))
