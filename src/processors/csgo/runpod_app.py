import sys
# sys.path.append("../")
sys.path.append("./")
sys.path.insert(0, '/workspace/runpod-sdxl-worker/src/processors/csgo/')

import torch
from torchvision import transforms
from ip_adapter.utils import BLOCKS as BLOCKS
from ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
from ip_adapter.utils import resize_content
import cv2
import numpy as np
import random
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,

)
from ip_adapter import CSGO
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
# image_encoder_path = "h94/IP-Adapter/sdxl_models/image_encoder"
image_encoder_path = "h94/IP-Adapter"

csgo_ckpt ='InstantX/CSGO/csgo_4_32.bin'
pretrained_vae_name_or_path ='madebyollin/sdxl-vae-fp16-fix'
# controlnet_path = "TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic"
controlnet_path = "./TTPLanet_SDXL_Controlnet_Tile_Realistic/"

weight_dtype = torch.float16




vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path,torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16,use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
    vae=vae
)
pipe.enable_vae_tiling()
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

target_content_blocks = BLOCKS['content']
target_style_blocks = BLOCKS['style']
controlnet_target_content_blocks = controlnet_BLOCKS['content']
controlnet_target_style_blocks = controlnet_BLOCKS['style']

csgo = CSGO(pipe, image_encoder_path, csgo_ckpt, device, num_content_tokens=4, num_style_tokens=32,
            target_content_blocks=target_content_blocks, target_style_blocks=target_style_blocks,
            controlnet_adapter=True,
            controlnet_target_content_blocks=controlnet_target_content_blocks,
            controlnet_target_style_blocks=controlnet_target_style_blocks,
            content_model_resampler=True,
            style_model_resampler=True,
            )

MAX_SEED = np.iinfo(np.int32).max

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed





def get_example():
    case = [
        [
            "./assets/img_0.png", #content_image_pil 

            './assets/img_1.png', #style_image_pil
            "Image-Driven Style Transfer", #target
            "there is a small house with a sheep statue on top of it", #prompt
            1.0, #scale_c_controlnet
            0.6, #scale_c
            1.0, #scale_s
        ],
        [
         None,
         './assets/img_1.png',
            "Text-Driven Style Synthesis",
         "a cat",
            1.0,
         0.01,
            1.0
         ],
        [
            None,
            './assets/img_2.png',
            "Text-Driven Style Synthesis",
            "a building",
            0.5,
            0.01,
            1.0
        ],
        [
            "./assets/img_0.png",
            './assets/img_1.png',
            "Text Edit-Driven Style Synthesis",
            "there is a small house",
            1.0,
            0.4,
            1.0
        ],
    ]
    return case


def run_for_examples(content_image_pil,style_image_pil,target, prompt,scale_c_controlnet, scale_c, scale_s):
    return create_image(
        content_image_pil=content_image_pil,
        style_image_pil=style_image_pil,
        prompt=prompt,
        scale_c_controlnet=scale_c_controlnet,
        scale_c=scale_c,
        scale_s=scale_s,
        guidance_scale=7.0,
        num_samples=3,
        num_inference_steps=50,
        seed=42,
        target=target,
    )
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def create_image(content_image_pil,
                 style_image_pil,
                 prompt,
                 scale_c_controlnet,
                 scale_c,
                 scale_s,
                 guidance_scale,
                 num_samples,
                 num_inference_steps,
                 seed,
                 target="Image-Driven Style Transfer",
):

    if content_image_pil is None:
        content_image_pil = Image.fromarray(
            np.zeros((1024, 1024, 3), dtype=np.uint8)).convert('RGB')

    if prompt is None or prompt == '':
        with torch.no_grad():
            inputs = blip_processor(content_image_pil, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs)
            prompt = blip_processor.decode(out[0], skip_special_tokens=True)
    width, height, content_image = resize_content(content_image_pil)
    style_image = style_image_pil
    neg_content_prompt='text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry'
    if target =="Image-Driven Style Transfer":
        with torch.no_grad():
            images = csgo.generate(pil_content_image=content_image, pil_style_image=style_image,
                                   prompt=prompt,
                                   negative_prompt=neg_content_prompt,
                                   height=height,
                                   width=width,
                                   content_scale=scale_c,
                                   style_scale=scale_s,
                                   guidance_scale=guidance_scale,
                                   num_images_per_prompt=num_samples,
                                   num_inference_steps=num_inference_steps,
                                   num_samples=1,
                                   seed=seed,
                                   image=content_image.convert('RGB'),
                                   controlnet_conditioning_scale=scale_c_controlnet,
                                   )

    elif target =="Text-Driven Style Synthesis":
        content_image = Image.fromarray(
            np.zeros((1024, 1024, 3), dtype=np.uint8)).convert('RGB')
        with torch.no_grad():
            images = csgo.generate(pil_content_image=content_image, pil_style_image=style_image,
                                   prompt=prompt,
                                   negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                                   height=height,
                                   width=width,
                                   content_scale=scale_c,
                                   style_scale=scale_s,
                                   guidance_scale=7,
                                   num_images_per_prompt=num_samples,
                                   num_inference_steps=num_inference_steps,
                                   num_samples=1,
                                   seed=42,
                                   image=content_image.convert('RGB'),
                                   controlnet_conditioning_scale=scale_c_controlnet,
                                   )
    elif target =="Text Edit-Driven Style Synthesis":

        with torch.no_grad():
            images = csgo.generate(pil_content_image=content_image, pil_style_image=style_image,
                                   prompt=prompt,
                                   negative_prompt=neg_content_prompt,
                                   height=height,
                                   width=width,
                                   content_scale=scale_c,
                                   style_scale=scale_s,
                                   guidance_scale=guidance_scale,
                                   num_images_per_prompt=num_samples,
                                   num_inference_steps=num_inference_steps,
                                   num_samples=1,
                                   seed=seed,
                                   image=content_image.convert('RGB'),
                                   controlnet_conditioning_scale=scale_c_controlnet,
                                   )

    return [transforms.ToPILImage()(img) for img in images]


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_cv2

def run_first_example():
    example = get_example()[0]
    content_image_pil = Image.open(example[0])
    style_image_pil = Image.open(example[1])
    target = example[2]
    prompt = example[3]
    scale_c_controlnet = example[4]
    scale_c = example[5]
    scale_s = example[6]
    return run_for_examples(content_image_pil,style_image_pil,target, prompt,scale_c_controlnet, scale_c, scale_s)
