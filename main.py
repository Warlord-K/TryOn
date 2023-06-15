from utils.model import load, generate
import click

@click.command()
@click.option('-i','--image_path',type = str, default = None, help = "Path to the Input Image")
@click.option('-c','--cloth_path',type = str, default = None, help = "Path to the Cloth Image")
@click.option('-o','--output_dir',type = str, default = '', help = "Path To the Output Directory Where Preview will be saved")
@click.option('-p','--prompt',type = str, default = None, help = "Prompt for Image Editing (Optional)")
def main(image_path, cloth_path, output_dir, prompt):
    using_prompt = True if prompt else False
    extractor, model, pipe = load(using_prompt)
    gen = generate(image_path, extractor, model, pipe, cloth_path, prompt)
    gen.save(f"{output_dir}/generated_preview.png")
    click.secho(f'Preview Generated Successfully at {output_dir}/generated_preview.png', fg="green", bold=True)

if __name__ == "__main__":
    main()
