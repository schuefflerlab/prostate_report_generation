import math
from pathlib import Path
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import openslide
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random


def attention_map(output: str, img_p: str, coords: npt.ArrayLike, attention_weights: npt.ArrayLike):
    attention_weights = attention_weights[:len(coords)]
    img = openslide.open_slide(img_p)
    lv0_w, lv0_h = img.dimensions
    thumbnail = img.get_thumbnail((3840, 2160)).convert("RGB")
    img = np.array(thumbnail)

    w, h = thumbnail.size
    aspect_ratio = w / h
    scale_w = lv0_w / w
    scale_h = lv0_h / h

    coords_f = coords.astype(np.float32)
    coords_f //= [scale_w, scale_h]

    scaled_attention_weights = (attention_weights - attention_weights.min()) / (
                attention_weights.max() - attention_weights.min())

    # print(f"{scaled_attention_weights.min()}, {scaled_attention_weights.max()}, {scaled_attention_weights.mean()}")
    coord_size = math.ceil(512 / scale_w)

    attn_map = np.zeros_like(img[:, :, 0], dtype=np.float32)
    mask = np.ones_like(img[:, :, 0])

    for c, a in zip(coords_f, scaled_attention_weights):
        x, y = c
        x, y = int(x), int(y)
        attn_map[y:y+coord_size, x:x+coord_size] = float(a)
        mask[y:y+coord_size, x:x+coord_size] = 0

    fig_height = 6
    fig_width = fig_height * aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(img)

    masked_attention_map = np.ma.masked_where(mask, attn_map)
    attention_map_image = ax.imshow(masked_attention_map, cmap='viridis', alpha=0.5, interpolation='nearest')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(attention_map_image, cax=cax)
    cb.set_label('Attention weight')

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{output}/heatmaps/{Path(img_p).stem}.jpg", dpi=300)
    plt.close()

    plt.hist(scaled_attention_weights, bins=32)
    plt.savefig(f"{output}/histograms/{Path(img_p).stem}_weights.jpg", dpi=200)
    plt.close()

    # plt.imshow(attn_map, cmap='viridis')
    # plt.savefig(f"{output}/attention_maps/{Path(img_p).stem}.jpg", dpi=300)
    # plt.close()