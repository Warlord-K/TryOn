from click.testing import CliRunner
from main import main


def test_cli():
    runner = CliRunner()
    result = runner.invoke(main, ["-i", "image.jpeg", "-c", "cloth.jpg"])
    assert result.exit_code == 0
    assert result.output.split("\n")[-2] == 'Preview Generated Successfully at /generated_preview.png'
