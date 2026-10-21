"""The only runtime model: pinned PP-OCRv6 small, recognition only, CPU."""
import hashlib
import json

import numpy as np

from .data import ROOT


def load_model():
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import OCRVersion,ModelType
    metadata=json.loads((ROOT/'research/model.json').read_text(encoding='utf-8'))
    weight=ROOT/'models/rapidocr'/metadata['filename']
    def check_weight():
        if weight.exists() and hashlib.sha256(weight.read_bytes()).hexdigest()!=metadata['sha256']:
            raise RuntimeError('Recognizer checksum differs from the tested model')
    check_weight()
    engine=RapidOCR(params={
        'Global.use_det':False,'Global.use_cls':False,'Global.text_score':0.0,
        'Global.model_root_dir':str(ROOT/'models/rapidocr'),'Global.log_level':'warning',
        'Global.use_preprocess_img':False,'Rec.ocr_version':OCRVersion.PPOCRV6,
        'Rec.model_type':ModelType.SMALL,'EngineConfig.onnxruntime.intra_op_num_threads':2,
        'EngineConfig.onnxruntime.inter_op_num_threads':1,
    })
    if not weight.is_file():raise RuntimeError('Expected recognizer weights were not materialized')
    check_weight()
    def predict(image):
        # RapidOCR ndarray input is BGR; Pillow input is converted from RGB.
        result=engine(np.asarray(image.convert('RGB'))[:,:,::-1].copy())
        texts=result.txts if result.txts is not None else ()
        scores=result.scores if result.scores is not None else ()
        return ''.join(texts),float(np.mean(scores)) if len(scores) else None
    return predict
