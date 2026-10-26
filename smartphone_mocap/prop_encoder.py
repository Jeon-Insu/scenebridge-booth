"""DINOv2 embeddings for prop instance retrieval.

The backbone stays frozen so a newly generated asset is searchable as soon as its views are
embedded — a trained ReID head would need retraining before it could recognise a new prop.
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_ID = 'facebook/dinov2-small'
# The SGLang container mounts the standard HF cache and owns its lock directory as root, so
# downloads from this venv fail there. huggingface_hub fixes its paths when it is first
# imported — and RF-DETR imports it before this module — so the directory is passed per call
# rather than through the environment.
CACHE_DIR = Path(os.environ.get('HF_HUB_CACHE') or Path.home() / '.cache/huggingface-gist/hub')
# ImageNet mean in 0-255; transparent pixels are filled with it rather than black, since a
# black background is itself a strong signal that distorts the embedding.
FILL_RGB = (124, 116, 104)


def flatten_alpha(image: Image.Image) -> Image.Image:
    """Composite a cutout onto the mean-pixel background."""
    if image.mode not in ('RGBA', 'LA') and 'transparency' not in image.info:
        return image.convert('RGB')
    rgba = image.convert('RGBA')
    background = Image.new('RGB', rgba.size, FILL_RGB)
    background.paste(rgba, mask=rgba.split()[3])
    return background


class PropEncoder:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=str(CACHE_DIR))
        self.model = (AutoModel.from_pretrained(model_id, cache_dir=str(CACHE_DIR))
                      .to(self.device).eval())

    def encode(self, images) -> np.ndarray:
        """L2-normalised CLS embeddings, one row per image."""
        prepared = [flatten_alpha(Image.open(i) if isinstance(i, (str, bytes)) or hasattr(i, '__fspath__') else i)
                    for i in images]
        inputs = self.processor(images=prepared, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        vectors = outputs.last_hidden_state[:, 0].float().cpu().numpy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-8, None)
