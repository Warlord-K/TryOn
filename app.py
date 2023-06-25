from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
from utils.model import load, generate
from utils.scraper import extract_link
import tempfile

LOADED = False
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "route working"}


@app.get("/generate")
async def generate_(image_path: str, cloth_path: str = None, prompt: str = None):
    """
    Generate Image.

    Request Body
    request = {
        "image" : Input Image URL
        "cloth" : Cloth Image URL
        "prompt" : Prompt, In case example image is not provided
    }

    Return Body:
    {
    gen: Generated Image
    }
    """
    using_prompt = True if prompt else False
    extractor, model, pipe = load(using_prompt)
    image_url = extract_link(image_path)
    cloth_url = extract_link(cloth_path)
    image_path = image_url if image_url else image_path
    cloth_path = cloth_url if cloth_url else cloth_path
    gen = generate(image_path, extractor, model, pipe, cloth_path, prompt)
    temp_file = tempfile.mkstemp(suffix=".jpg")
    gen.save(temp_file[-1])
    return FileResponse(temp_file[-1])


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
