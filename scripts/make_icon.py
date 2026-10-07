"""Render icon.png (512x512): a waveform on a rounded dark tile. Needs numpy + ffmpeg."""

import subprocess
from pathlib import Path

import numpy as np

S = 512
img = np.zeros((S, S, 4), dtype=np.uint8)
y, x = np.mgrid[0:S, 0:S]
r, pad = 110, 16
cx = np.clip(x, pad + r, S - pad - r)
cy = np.clip(y, pad + r, S - pad - r)
tile = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
t = (x + y) / (2 * S)
img[tile, 0] = (20 + 30 * t[tile]).astype(np.uint8)
img[tile, 1] = (18 + 10 * t[tile]).astype(np.uint8)
img[tile, 2] = (40 + 60 * t[tile]).astype(np.uint8)
img[tile, 3] = 255

bars = 13
heights = [0.18, 0.32, 0.5, 0.36, 0.62, 0.82, 0.95, 0.78, 0.55, 0.7, 0.42, 0.28, 0.16]
bw, gap = 22, 10
x0 = (S - (bars * bw + (bars - 1) * gap)) // 2
for i, h in enumerate(heights):
    half = int(h * 300 / 2)
    left = x0 + i * (bw + gap)
    mask = (x >= left) & (x < left + bw) & (np.abs(y - S // 2) <= half)
    k = i / (bars - 1)
    img[mask] = [int(255 * (1 - k) + 120 * k), int(140 * (1 - k) + 200 * k), int(60 * (1 - k) + 255 * k), 255]

out = Path(__file__).resolve().parent.parent / "icon.png"
subprocess.run(
    ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{S}x{S}", "-i", "-", str(out)],
    input=img.tobytes(),
    check=True,
)
print(out)
