from fastapi import FastAPI, Request
import uvicorn
from utils.model import load, generate

LOADED = False
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "route working"}

@app.get("/generate")
async def add(request : Request):
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
    image_path = request.get("image", None)
    if not image_path:
        return {"Error": "No Image Provided"}
    cloth_path = request.get("cloth", None)
    prompt = request.get("prompt", None)
    using_prompt = True if prompt else False
    if not LOADED:
        extractor, model, pipe = load(using_prompt)
    gen = generate(image_path, extractor, model, pipe, cloth_path, prompt)

    return {"gen": gen}

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')