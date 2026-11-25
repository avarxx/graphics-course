#!/usr/bin/env python3
"""
Builds `source.png` for 3D LUT creation: screenshots of the renderer (taken with color grading
off) stacked vertically, plus an "unwrapped" identity 32x32x32 LUT (a grid of 32 tiles of
32x32 pixels: x = red, y = green, tile = blue) appended at the bottom. The identity grid
guarantees that every cell of the RGB cube gets a color, the screenshots weight the colors that
actually occur in the game. Grade the result in any graphics editor with GLOBAL color operations
only (curves, levels, color balance, hue/saturation, ...), save it as `graded.png` of the same
size into a folder next to this script, and restart the app.

usage: make_source.py out.png screenshot1.png [screenshot2.png ...] [--width 640]
"""
import sys
from PIL import Image

N = 32


def identity_grid(tiles_x=8):
    tiles_y = N // tiles_x
    img = Image.new("RGB", (tiles_x * N, tiles_y * N))
    px = img.load()
    for b in range(N):
        tx, ty = b % tiles_x, b // tiles_x
        for g in range(N):
            for r in range(N):
                v = lambda c: c * 255 // (N - 1)
                px[tx * N + r, ty * N + g] = (v(r), v(g), v(b))
    return img


def main():
    args = sys.argv[1:]
    width = 640
    if "--width" in args:
        i = args.index("--width")
        width = int(args[i + 1])
        del args[i:i + 2]
    out, shots = args[0], args[1:]
    parts = []
    for s in shots:
        im = Image.open(s).convert("RGB")
        h = round(im.height * width / im.width)
        parts.append(im.resize((width, h), Image.BOX))
    grid = identity_grid()
    total_h = sum(p.height for p in parts) + grid.height
    result = Image.new("RGB", (max(width, grid.width), total_h))
    y = 0
    for p in parts:
        result.paste(p, (0, y))
        y += p.height
    result.paste(grid, (0, y))
    result.save(out)
    print(f"{out}: {result.width}x{result.height}")


if __name__ == "__main__":
    main()
