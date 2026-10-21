import io
import json
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
import requests
from PIL import Image

from .data import ROOT


def fetch_preview(repo_id='project-sloth/captcha-images', split='test', limit=16):
    """Download a bounded viewer sample, not a training/evaluation benchmark."""
    if not 1 <= limit <= 100:
        raise ValueError('Choose 1..100 preview rows')
    lock = json.loads((ROOT / 'research/hf_lock.json').read_text(encoding='utf-8'))
    expected = lock['datasets'][repo_id]['revision']
    with requests.Session() as session:
        current = session.get('https://huggingface.co/api/datasets/'+repo_id,timeout=30)
        current.raise_for_status()
        # Viewer serves current revision. Abort rather than label newer data with an old SHA.
        if current.json()['sha'] != expected:
            raise ValueError('Dataset changed; review metadata and update lock before downloading')
        response = session.get('https://datasets-server.huggingface.co/rows',params={
            'dataset':repo_id,'config':'default','split':split,'offset':0,'length':limit},timeout=60)
        response.raise_for_status()
        records = response.json()['rows']
        destination = ROOT / 'data/hf_samples' / repo_id.replace('/','--')
        destination.mkdir(parents=True, exist_ok=True)
        rows = []
        for record in records:
            row = record['row']
            label = row.get('solution', row.get('text', row.get('label')))
            if not isinstance(label,str):
                raise ValueError('Expected string label; refusing to lose leading zeros')
            url = row['image']['src']
            host = urlsplit(url).hostname or ''
            if not (host.endswith('.huggingface.co') or host == 'huggingface.co' or host.endswith('.hf.co')):
                raise ValueError('Unexpected image host')
            image_response = session.get(url,timeout=45)
            image_response.raise_for_status()
            with Image.open(io.BytesIO(image_response.content)) as image:
                path = destination / f'{split}_{record["row_idx"]:06d}.png'
                image.convert('RGB').save(path)
            rows.append(dict(image_id=path.stem,path=str(path),label=label,label_status='verified',
                             source=repo_id,split=split,revision=expected))
    frame = pd.DataFrame(rows)
    frame.to_csv(destination/'manifest.csv',index=False,encoding='utf-8')
    return frame
