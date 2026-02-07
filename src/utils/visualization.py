from typing import Any

import matplotlib.pyplot as plt


def show_image(tensor, title: str = "") -> None:
    """简单的可视化辅助函数。"""
    img = tensor.detach().cpu()
    if img.dim() == 3 and img.size(0) in (1, 3):
        img = img.permute(1, 2, 0)
    plt.imshow(img, cmap="gray" if img.ndim == 2 else None)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


