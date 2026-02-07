from pathlib import Path
from typing import Generator, Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_images_recursive(root: Path) -> Generator[Path, None, None]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


