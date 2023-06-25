from fastapi import FastAPI
from fastapi.responses import FileResponse
import uvicorn
from utils.model import load, generate
import tempfile
LOADED = False
app = FastAPI()
@app.get("/")
async def root():
    return {"message": "route working"}

@app.get("/generate")
async def generate_(image_path : str, cloth_path : str = None, prompt : str = None):
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
        LOADED = True
    gen = generate(image_path, extractor, model, pipe, cloth_path, prompt)
    temp_file = tempfile.mkstemp(suffix = '.jpg')
    gen.save(temp_file[-1])
    return FileResponse(temp_file[-1])

if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')