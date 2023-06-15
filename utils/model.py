"""
This Module contains funstions for loading the segmentation model and inpainting models, and editing top using a example image or text prompt.

"""

# Imports 
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torch.nn as nn
import torch

# Functions
def load_seg(model_card = "mattmdjaga/segformer_b2_clothes"):
    """
    Load The Segmentation Extractor and Model.
    
    Arguements:
    model_card: HuggingFace Model Card. Default: mattmdjaga/segformer_b2_clothes
    Returns:
    extractor: Feature Extractor
    model: Segformer Model For Segmentation
    """
    extractor = AutoFeatureExtractor.from_pretrained(model_card)
    model = SegformerForSemanticSegmentation.from_pretrained(model_card)
    return extractor, model

def load_inpainting(using_prompt = False):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if using_prompt:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16
        )
        pipe = pipe.to(device)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            "Fantasy-Studio/Paint-by-Example",
            torch_dtype=torch.float16,
        )
        pipe = pipe.to(device)
    return pipe

def generate_mask(image_name, extractor, model):
    image = Image.open(image_name)
    inputs = extractor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]
    pred_seg[pred_seg != 4] = 0
    pred_seg[pred_seg == 4] = 1
    pred_seg = pred_seg.to(dtype = torch.float32)
    # pred_seg = pred_seg.unsqueeze(dim = 0)
    mask = to_pil_image(pred_seg)
    return image, mask

def generate_image(image, mask, pipe, example_name = None, prompt = None):
    if example_name:
        example = Image.open(example_name)
        gen = pipe(image=image, mask_image = mask, example_image = example).images[0]
    elif prompt:
        gen = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    return image, mask, gen

def load(using_prompt = False):
    extractor, model = load_seg()
    pipe = load_inpainting(using_prompt)
    return extractor, model, pipe

def generate(image_name, extractor, model, pipe, example_name = None, prompt = None):
    image, mask = generate_mask(image_name, extractor, model)
    res = int(mask.size[1] * 512/mask.size(0))
    image, mask, gen = generate_image(image, mask, pipe, example_name, prompt)
    return gen.resize((512, res))




