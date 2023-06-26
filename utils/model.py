"""
This Module contains funstions for loading the segmentation model and inpainting models, and editing top using a example image or text prompt.

"""

# Imports
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torch
import urllib.request


# Functions
def load_seg(model_card: str = "mattmdjaga/segformer_b2_clothes"):
    """
    Load The Segmentation Extractor and Model.

    Parameters:
    model_card: HuggingFace Model Card. Default: mattmdjaga/segformer_b2_clothes

    Returns:
    extractor: Feature Extractor
    model: Segformer Model For Segmentation
    """
    extractor = AutoFeatureExtractor.from_pretrained(model_card)
    model = SegformerForSemanticSegmentation.from_pretrained(model_card)
    return extractor, model


def load_inpainting(using_prompt: bool = False, fast: bool = False):
    """
    Load Inpaining Model.

    Parameters:
    using_prompt: If using a prompt based inpainting model or image based inpainting model. Default: False

    Returns:
    pipe: Diffusion Pipeline mounted onto the device
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if using_prompt:
        if fast:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                revision="fp16",
                torch_dtype=torch.float16,
            )
        else:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float32,
            )
    else:
        if fast:
            pipe = DiffusionPipeline.from_pretrained(
                "Fantasy-Studio/Paint-by-Example",
                torch_dtype=torch.float16,
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(
                "Fantasy-Studio/Paint-by-Example",
                torch_dtype=torch.float32,
            )
    pipe = pipe.to(device)
    return pipe


def generate_mask(image_name: str, extractor, model):
    """
    Generate mask using Image Path and Segmentation Model.

    Parameters:
    image_name: Path to Input Image
    extractor: Feature Extractor
    model: Segmentation Model

    Returns:
    image: PIL Image of Input Image
    mask: PIL Image of Generated Mask
    """
    try:
        image = Image.open(image_name)
    except Exception as e:
        image = Image.open(urllib.request.urlopen(image_name))
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
    pred_seg = pred_seg.to(dtype=torch.float32)
    # pred_seg = pred_seg.unsqueeze(dim = 0)
    mask = to_pil_image(pred_seg)
    return image, mask


def generate_image(image, mask, pipe, example_name=None, prompt=None):
    """
    Generate Edited Image. Uses Example Image or Prompt.

    Parameters:
    image: PIL Image of The Image to Edit.
    mask: PIL Image of the Mask.
    pipe: DiffusionPipeline
    example_name: Path to Image of the cloth.
    prompt: Editing Prompt, if not using Example Image.

    Returns:
    image: PIL Image of Input Image
    mask: PIL Image of Generated Mask
    gen: PIL Image of Generated Preview
    """
    if example_name:
        try:
            example = Image.open(example_name)
        except Exception as e:
            example = Image.open(urllib.request.urlopen(example_name))
        gen = pipe(
            image=image.resize((512, 512)),
            mask_image=mask.resize((512, 512)),
            example_image=example.resize((512, 512)),
        ).images[0]
    elif prompt:
        gen = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    else:
        gen = None
        print("Neither Example Image nor Prompt provided.")
    return image, mask, gen


def load(using_prompt=False):
    """
    Loads Segmentation and Inpainting Model.

    Parameters:
    using_prompt: If using a prompt based inpainting model or image based inpainting model. Default: False

    Returns:
    extractor: Feature Extractor
    model: Segformer Model For Segmentation
    pipe: Diffusion Pipeline loaded onto the device
    """
    extractor, model = load_seg()
    pipe = load_inpainting(using_prompt)
    return extractor, model, pipe


def generate(image_name, extractor, model, pipe, example_name=None, prompt=None):
    """
    Generate Preview.

    Parameters:
    image_name: Path to Input Image
    extractor: Feature Extractor
    model: Segmentation Model
    pipe: DiffusionPipeline
    example_name: Path to Image of the cloth.
    prompt: Editing Prompt, if not using Example Image.

    Returns:
    gen: PIL Image of Generated Preview
    """
    image, mask = generate_mask(image_name, extractor, model)
    res = int(mask.size[1] * 512 / mask.size[0])
    image, mask, gen = generate_image(image, mask, pipe, example_name, prompt)
    return gen.resize((512, res))
