from click.testing import CliRunner
from main import main


def test_cli():
    runner = CliRunner()
    result = runner.invoke(main, ["-i", "image.jpg", "-c", "cloth.jpg"])
    assert result.exit_code == 0
    assert result.output == "Preview Generated Successfully at /generated_preview.png\n"
