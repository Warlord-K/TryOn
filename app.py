from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
from utils.model import load, generate, load_seg, load_inpainting
from utils.scraper import extract_link
import tempfile
from typing import Optional

LOADED = False
app = FastAPI()

extractor, model = load_seg()
prompt_pipe = load_inpainting(using_prompt=True)
cloth_pipe = load_inpainting()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Body(BaseModel):
    image_path: str
    cloth_path: str
    prompt: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "route working"}


@app.post("/generate")
async def generate_(body: Body):
    prompt = body.prompt
    image_path = body.image_path
    cloth_path = body.cloth_path
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
    image_url = extract_link(image_path)
    cloth_url = extract_link(cloth_path)
    image_path = image_url if image_url else image_path
    cloth_path = cloth_url if cloth_url else cloth_path
    if using_prompt:
        gen = generate(image_path, extractor, model, prompt_pipe, cloth_path, prompt)
    else:
        gen = generate(image_path, extractor, model, cloth_pipe, cloth_path, prompt)
    temp_file = tempfile.mkstemp(suffix=".jpg")
    gen.save(temp_file[-1])
    return FileResponse(temp_file[-1])


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")
