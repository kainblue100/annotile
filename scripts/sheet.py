"""Contact sheets for the Blender stills."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"E:\annotile\results\figures\blender")
DEST = Path(r"E:\annotile\results\figures")
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 32)


def sheet(kind: str, name: str) -> None:
    tiles = [Image.open(ROOT / f"P{index}_{kind}.png").convert("RGB") for index in range(1, 9)]
    width, height = tiles[0].size
    pad = 10
    label = 42
    columns, rows = 4, 2
    canvas = Image.new("RGB", (columns * width + (columns + 1) * pad, rows * (height + label) + (rows + 1) * pad), (246, 244, 239))
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(tiles):
        row, column = divmod(index, columns)
        x = pad + column * (width + pad)
        y = pad + row * (height + label + pad)
        draw.text((x + 6, y + 4), f"P{index + 1}", fill=(28, 25, 23), font=FONT)
        canvas.paste(image, (x, y + label))
    canvas.save(DEST / name)
    print("SHEET", name, canvas.size)


if __name__ == "__main__":
    sheet("cube", "blender_volume.png")
    sheet("close", "blender_closeup.png")
