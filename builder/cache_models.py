# builder/model_fetcher.py

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL, ControlNetModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import os

def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    os.system('apt-get install git-lfs')
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }

    # CSGO models

    #base
    fetch_pretrained_model(StableDiffusionXLPipeline,
                                  "stabilityai/stable-diffusion-xl-base-1.0", **common_args)

    #vae
    fetch_pretrained_model(AutoencoderKL,
                                  "madebyollin/sdxl-vae-fp16-fix")

    #control net
    os.system('git clone https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic')
    os.system('mv TTPLanet_SDXL_Controlnet_Tile_Realistic/TTPLANET_Controlnet_Tile_realistic_v2_fp16.safetensors TTPLanet_SDXL_Controlnet_Tile_Realistic/diffusion_pytorch_model.safetensors')

    fetch_pretrained_model(ControlNetModel,
                                  "./TTPLanet_SDXL_Controlnet_Tile_Realistic/", **common_args)
    

    #ip adapter
    os.system('git clone https://huggingface.co/InstantX/CSGO')

    
    #image_encoder
    CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="sdxl_models/image_encoder")

    #Check if needed
    BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
  
    return pipe


if __name__ == "__main__":
    get_diffusion_pipelines()
