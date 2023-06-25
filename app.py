from fastapi import FastAPI
import uvicorn
from utils.model import load, generate

LOADED = False
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "route working"}

@app.get("/generate")
async def generate(image_path : str, cloth_path : str = None, prompt : str = None):
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
    if not LOADED:
        extractor, model, pipe = load(using_prompt)
    gen = generate(image_path, extractor, model, pipe, cloth_path, prompt)

    return {"gen": gen}

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')