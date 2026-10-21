import hashlib
import io
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
# Manually selected on the supplied 1080x2220 layout; verify in notebook 00.
# This is a fixed experimental ROI, not a general screen detector.
DEFAULT_ROI = (0.10, 0.24, 0.93, 0.36)


def import_images(source, destination=None):
    source = Path(source)
    destination = Path(destination or ROOT / 'data/local')
    destination.mkdir(parents=True, exist_ok=True)
    found = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    for src in found:
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        shutil.copy2(src, destination / (digest[:16] + src.suffix.lower()))
    return len(found)


def image_manifest(directory=None):
    directory = Path(directory or ROOT / 'data/local')
    rows = []
    for p in sorted(directory.glob('*')):
        if p.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        try:
            raw = p.read_bytes()
            with Image.open(io.BytesIO(raw)) as im:
                im.load()
                rows.append(dict(image_id=p.stem, path=str(p), sha256=hashlib.sha256(raw).hexdigest(),
                                 width=im.width, height=im.height, bytes=len(raw), mode=im.mode, error=''))
        except Exception as exc:
            rows.append(dict(image_id=p.stem, path=str(p), error=type(exc).__name__))
    return pd.DataFrame(rows)


def crop_image(path, roi=DEFAULT_ROI, variant='rgb'):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
    if roi is not None:
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError('ROI must be normalized x1,y1,x2,y2 within [0,1]')
        image = image.crop((int(x1 * image.width), int(y1 * image.height),
                            int(x2 * image.width), int(y2 * image.height)))
    if variant == 'gray':
        image = ImageOps.grayscale(image).convert('RGB')
    elif variant != 'rgb':
        raise ValueError('Unknown preprocessing variant')
    return image


def png_bytes(image):
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def ensure_labels(manifest, path=None):
    path = Path(path or ROOT / 'data/labels.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() else pd.DataFrame(
        columns=['image_id', 'label', 'label_status', 'note'])
    if old.image_id.duplicated().any():
        raise ValueError('Duplicate image_id in labels.csv')
    known = set(old.image_id)
    additions = [dict(image_id=i, label='', label_status='unlabeled', note='')
                 for i in manifest.image_id if i not in known]
    result = pd.concat([old, pd.DataFrame(additions)], ignore_index=True)
    result.to_csv(path, index=False, encoding='utf-8')
    return result


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
