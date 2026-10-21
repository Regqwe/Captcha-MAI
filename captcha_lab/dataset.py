"""Download pinned full archives and materialize labels, images and holdouts."""
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import tarfile

import numpy as np
import pandas as pd
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'datasets/captcha-images'
REVISION = 'eeaf2b6ec9086645f270f7e2aaa9ea90730683de'
ARCHIVES = {
    'train': (6000, '7a8c388a990aa5a5f04b3e2503f87ecc2a0f8da1fdb70de368298e218490eb55'),
    'validation': (2000, '68d339c89403682ce2844d2d94524c15afcf990c057cba055dc6a985de42ea91'),
    'test': (2000, '730bf94a1b857caa1de00bc7b5d9840c25bcd9767718f91ba9f8a3d5fd2a1912'),
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare():
    DATASET.mkdir(parents=True, exist_ok=True)
    all_rows = []
    archives = []
    for split, (count, expected_sha) in ARCHIVES.items():
        archive = DATASET / (split + '.tar.gz')
        prior_download = ROOT / 'data/benchmark/archives' / archive.name
        if not archive.exists() and prior_download.exists():
            shutil.copy2(prior_download, archive)
        if not archive.exists():
            url = f'https://huggingface.co/datasets/project-sloth/captcha-images/resolve/{REVISION}/data/{split}.tar.gz'
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                part = archive.with_suffix('.part')
                with part.open('wb') as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
                assert digest(part.read_bytes()) == expected_sha, 'Archive checksum mismatch'
                part.replace(archive)
        assert digest(archive.read_bytes()) == expected_sha, 'Archive checksum mismatch'
        rows = []
        with tarfile.open(archive) as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                match = re.fullmatch(r'(\d{6})\.([a-f0-9]{32})\.jpg', name)
                if not match or member.name != f'{split}/{name}':
                    raise ValueError('Unexpected archive member; no unrestricted extraction')
                raw = tar.extractfile(member).read()
                with Image.open(io.BytesIO(raw)) as image:
                    image = image.convert('RGB')
                    pixel_hash = digest(str(image.size).encode() + image.tobytes())
                    width, height = image.size
                relative = Path('data/benchmark/images') / split / name
                target = ROOT / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or digest(target.read_bytes()) != digest(raw):
                    target.write_bytes(raw)
                rows.append(dict(image_id=split+'_'+match[2], source_split=split, label=match[1],
                                 image_path=relative.as_posix(), sha256=digest(raw), pixel_sha256=pixel_hash,
                                 width=width, height=height, label_source='dataset filename six-digit prefix',
                                 label_status='dataset_provided', dataset_revision=REVISION))
        assert len(rows) == count
        frame = pd.DataFrame(rows).sort_values('image_id').reset_index(drop=True)
        frame.to_csv(DATASET / (split+'_labels.csv'), index=False)
        all_rows.extend(rows)
        archives.append(dict(split=split, images=len(rows), bytes=archive.stat().st_size, sha256=expected_sha))
        print('Materialized', split, len(rows), flush=True)

    complete = pd.DataFrame(all_rows)
    assert complete.image_id.is_unique
    assert complete.label.str.fullmatch(r'\d{6}').all()
    # Exclude all previously viewed 16 test rows, even if encoding changed.
    # The labels are explicitly frozen so fresh clones reproduce the same exclusion.
    previous_labels = ['001251','001335','002402','002769','002821','005491','005563','005768',
                       '006007','006402','006664','006949','008717','008947','009223','011166']
    development_pixels = set(complete.loc[complete.source_split.ne('test'),'pixel_sha256'])
    test = complete[complete.source_split.eq('test')].copy()
    excluded = test[test.label.isin(previous_labels) | test.pixel_sha256.isin(development_pixels)].copy()
    excluded['reason'] = np.where(excluded.label.isin(previous_labels),'viewed_in_previous_pilot','duplicate_development_pixels')
    holdout = test[~test.image_id.isin(excluded.image_id)].drop_duplicates('pixel_sha256').sort_values('image_id')
    excluded.to_csv(DATASET/'excluded_from_holdout.csv',index=False)
    holdout.to_csv(DATASET/'holdout_labels.csv',index=False)
    rng = np.random.default_rng(20261021)
    shared = holdout.iloc[np.sort(rng.choice(len(holdout),size=200,replace=False))]
    shared.to_csv(DATASET/'shared_200_labels.csv',index=False)
    by_pixel = complete.groupby('pixel_sha256').source_split.nunique()
    audit = dict(dataset='project-sloth/captcha-images', revision=REVISION, archives=archives,
                 total_images=len(complete), source_split_counts=complete.source_split.value_counts().to_dict(),
                 all_labels_six_digit_strings=True, leading_zero_labels=int(complete.label.str.startswith('0').sum()),
                 byte_duplicates=int(complete.sha256.duplicated().sum()),
                 pixel_duplicates=int(complete.pixel_sha256.duplicated().sum()),
                 cross_split_pixel_groups=int(by_pixel.gt(1).sum()), excluded_test_rows=len(excluded),
                 holdout_images=len(holdout), shared_images=len(shared), selection_seed=20261021,
                 same_label_cross_split_groups=int(complete.groupby('label').source_split.nunique().gt(1).sum()),
                 holdout_manifest_sha256=digest((DATASET/'holdout_labels.csv').read_bytes()),
                 shared_manifest_sha256=digest((DATASET/'shared_200_labels.csv').read_bytes()))
    (DATASET/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2),flush=True)
    return audit


if __name__ == '__main__':
    prepare()
