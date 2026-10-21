import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, DEFAULT_ROI, crop_image


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j-1] + (ca != cb)))
        previous = current
    return previous[-1]


def run_benchmark(manifest, recognizer, name, *, roi=DEFAULT_ROI, variants=('rgb',), repeats=3, warmup=True):
    if manifest.empty:
        raise ValueError('No images; import data in notebook 00')
    if repeats < 1:
        raise ValueError('repeats must be >= 1')
    # Initialization/download is outside this timer. First inference is reported separately.
    cold_ms = None
    if warmup:
        start = time.perf_counter()
        recognizer(crop_image(manifest.iloc[0].path, roi=roi, variant=variants[0]))
        cold_ms = (time.perf_counter() - start) * 1000
    rows = []
    for variant in variants:
        for item in manifest.itertuples():
            durations, answers, scores = [], [], []
            error, error_detail = '', ''
            for _ in range(repeats):
                try:
                    start = time.perf_counter()
                    image = crop_image(item.path, roi=roi, variant=variant)
                    prediction, score = recognizer(image)
                    durations.append((time.perf_counter() - start) * 1000)
                    answers.append(str(prediction))
                    if score is not None:
                        scores.append(float(score))
                except Exception as exc:
                    error = type(exc).__name__
                    error_detail = getattr(exc, 'safe_message', '')
                    break
            rows.append(dict(model=name, image_id=item.image_id, variant=variant,
                             prediction=answers[-1] if answers else '', score=scores[-1] if scores else None,
                             median_ms=float(np.median(durations)) if durations else None,
                             p95_ms=float(np.percentile(durations, 95)) if durations else None,
                             repeats_completed=len(answers), stable=bool(answers) and len(set(answers)) == 1,
                             error=error, error_detail=error_detail, roi=json.dumps(roi)))
    result = pd.DataFrame(rows)
    result.attrs['first_inference_ms'] = cold_ms
    result.attrs['timer_scope'] = 'crop + preprocessing + inference + decoding; excludes model load/download'
    return result


def evaluate(predictions, labels):
    if labels.image_id.duplicated().any():
        raise ValueError('Duplicate image_id in labels')
    verified = labels[(labels.label_status == 'verified') & labels.label.ne('')]
    rows = []
    for (model, variant), group in predictions.groupby(['model', 'variant']):
        joined = group.merge(verified[['image_id', 'label']], on='image_id', how='inner', validate='many_to_one')
        total = len(joined)
        exact = sum((r.prediction == r.label) and not r.error for r in joined.itertuples())
        chars = sum(len(r.label) for r in joined.itertuples())
        edits = sum(edit_distance(r.prediction if not r.error else '', r.label) for r in joined.itertuples())
        rows.append(dict(model=model, variant=variant, images=len(group), verified=total,
                         errors=int(group.error.ne('').sum()),
                         exact_match=exact / total if total else None,
                         cer=edits / chars if chars else None,
                         median_ms=group.median_ms.median(),
                         # This is a quantile of per-image medians, not pooled request p95.
                         p95_image_median_ms=group.median_ms.quantile(.95)))
    return pd.DataFrame(rows)


def save_run(frame, filename):
    target = ROOT / 'results' / filename
    target.parent.mkdir(exist_ok=True)
    frame.to_csv(target, index=False, encoding='utf-8')
    packages = {}
    for name in ['ddddocr','rapidocr','onnxruntime','torch','transformers','pillow','numpy']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    meta = dict(frame.attrs, platform=platform.platform(), processor=platform.processor(), packages=packages)
    target.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return target
