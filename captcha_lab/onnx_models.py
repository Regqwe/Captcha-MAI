import numpy as np
from .data import ROOT, png_bytes


def load_ddddocr(beta=False):
    import ddddocr
    model = ddddocr.DdddOcr(show_ad=False, beta=beta)
    def predict(image):
        return model.classification(png_bytes(image)), None
    return predict


def load_ppocr(size='tiny'):
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import OCRVersion, ModelType
    if size not in {'tiny', 'small'}:
        raise ValueError('Choose tiny or small')
    model = RapidOCR(params={
        'Global.use_det': False, 'Global.use_cls': False, 'Global.text_score': 0.0,
        'Global.model_root_dir': str(ROOT / 'models/rapidocr'),
        'Global.log_level': 'warning', 'Global.use_preprocess_img': False,
        'Rec.ocr_version': OCRVersion.PPOCRV6,
        'Rec.model_type': ModelType.TINY if size == 'tiny' else ModelType.SMALL,
        'EngineConfig.onnxruntime.intra_op_num_threads': 2,
        'EngineConfig.onnxruntime.inter_op_num_threads': 1,
    })
    def predict(image):
        # RapidOCR's ndarray input is BGR; PIL output is RGB.
        result = model(np.asarray(image)[:, :, ::-1].copy())
        texts = result.txts if result.txts is not None else ()
        scores = result.scores if result.scores is not None else ()
        return ''.join(texts), float(np.mean(scores)) if len(scores) else None
    return predict
