# Try On AI
Virtual Try On for Online Clothes. Have you ever wanted to Try On Clothes before buying them Online? Well this tool is made just for that! Just Provide One Photo and One URL to an Image of Cloth or Any Product URL from Flipkart, Amazon or Myntra and see the generated preview within seconds!
[Deployed On Huggingface](https://huggingface.co/spaces/Warlord-K/TryOn)

Sample Input:
![image](image.jpeg)

Sample Output:
![image](https://github.com/Warlord-K/TryOn/assets/95569637/63fe45bd-cf74-4595-873b-78ab45c5a57d)


## Models Used
* [**Segformer**](https://huggingface.co/mattmdjaga/segformer_b2_clothes): For Segmenting out original clothes from Image
* [**Paint by Example**](https://huggingface.co/Fantasy-Studio/Paint-by-Example): For Generating Preview using User Image, Cloth Image and Segmentation Mask from Segformer.
* [**Stable Diffusion Inpainting**](https://huggingface.co/runwayml/stable-diffusion-inpainting): For Generating Image Using Prompt Instead of Cloth Image.

## Modules Used

For Image Generation and Segmentation:
* Transformers
* Diffusers

For Scraping Cloth Image From URL:
* Beautiful Soup
* Selenium

For Creating Interface:
* Gradio

For Creating API Backend:
* FastAPI
* Uvicorn

For CLI:
* Click

For Tests:
* Pytest

For Linting:
* Pylint

For Formatting:
* Black

## Installation

* Clone the Repository
  * ```git clone https://github.com/Warlord-K/TryOn.git```
* Install Requirements
  * ```pip install -r requirements.txt```
* Run FastAPI Server
  * ```python app.py```
* Optionally Run the CLI
  * ```python3 main.py -i image.jpeg -c cloth.jpg```
