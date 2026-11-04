"""Put a transparent diorama render on a soft background, and make a before/after with the Game Boy frame."""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

render, frame_npy, out = sys.argv[1:4]
im = Image.open(render).convert("RGBA")
w, h = im.size
t = np.linspace(0, 1, h)[:, None, None]
bg = (np.array([24, 30, 44]) * (1 - t) + np.array([58, 66, 86]) * t).repeat(w, 1).astype(np.uint8)
bg = Image.fromarray(bg).convert("RGBA")
dio = Image.alpha_composite(bg, im).convert("RGB")
dio.save(out + "_diorama.png")

q = np.load(frame_npy)
gb = Image.fromarray((q * 85).astype(np.uint8)).resize((480, 432), Image.NEAREST).convert("RGB")
big = dio.resize((1280, 720), Image.LANCZOS)
W, H = 480 + 1280 + 3 * 60, 720 + 2 * 70
canvas = Image.new("RGB", (W, H), (24, 30, 44))
canvas.paste(gb, (60, 70 + (720 - 432) // 2))
canvas.paste(big, (60 + 480 + 60, 70))
d = ImageDraw.Draw(canvas)
font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 26)
d.text((60, 70 + (720 - 432) // 2 - 44), "Game Boy frame (160x144)", fill=(220, 224, 232), font=font)
d.text((600, 26), "the same frame as a voxel diorama (Blender, rendered on the CPU)", fill=(220, 224, 232), font=font)
canvas.save(out + "_before_after.png")
print("saved", out + "_diorama.png", out + "_before_after.png")
