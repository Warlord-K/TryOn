from utils.model import (
    load_seg,
    load,
    load_inpainting,
    generate_mask,
    generate_image,
    generate,
)
import PIL
from transformers.models.segformer.feature_extraction_segformer import (
    SegformerFeatureExtractor,
)
from transformers.models.segformer.modeling_segformer import (
    SegformerForSemanticSegmentation,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline,
)
from diffusers.pipelines.paint_by_example.pipeline_paint_by_example import (
    PaintByExamplePipeline,
)


def test_load_seg():
    extractor, model = load_seg()
    assert type(extractor) == SegformerFeatureExtractor
    assert type(model) == SegformerForSemanticSegmentation


def test_load_impainting_prompt():
    pipe = load_inpainting(True)
    assert type(pipe) == StableDiffusionInpaintPipeline


def test_load_impainting_example():
    pipe = load_inpainting()
    assert type(pipe) == PaintByExamplePipeline


def test_load():
    extractor, model, pipe = load()
    assert type(extractor) == SegformerFeatureExtractor
    assert type(model) == SegformerForSemanticSegmentation
    assert type(pipe) == PaintByExamplePipeline


def test_generate_mask():
    extractor, model = load_seg()
    image, mask = generate_mask("image.jpg", extractor, model)
    assert type(image) == PIL.JpegImagePlugin.JpegImageFile
    assert type(mask) == PIL.Image.Image

def test_generate_image_prompt():
    extractor, model, pipe = load(True)
    image, mask = generate_mask("image.jpg", extractor, model)
    image, mask, gen = generate_image(image, mask, pipe, prompt = "Blue Jacket")
    assert type(image) == PIL.JpegImagePlugin.JpegImageFile
    assert type(mask) == PIL.Image.Image
    assert type(gen) == PIL.Image.Image

def test_generate_image_example():
    extractor, model, pipe = load()
    image, mask = generate_mask("image.jpg", extractor, model)
    image, mask, gen = generate_image(image, mask, pipe, "cloth.jpg")
    assert type(image) == PIL.JpegImagePlugin.JpegImageFile
    assert type(mask) == PIL.Image.Image
    assert type(gen) == PIL.Image.Image

def test_generate():
    extractor, model, pipe = load()
    gen = generate("image.jpg", extractor, model, pipe, "cloth.jpg")
    assert type(gen) == PIL.Image.Image

    