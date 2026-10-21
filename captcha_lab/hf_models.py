"""Pinned safetensors weights with local inference code; no trust_remote_code.

Architecture/preprocessing adapted from Graf-J's MIT-licensed model repositories.
See THIRD_PARTY_NOTICES.md and research/hf_lock.json for source revisions.
"""
import json
import math
import string

import numpy as np

from .data import ROOT


def load_graf(repo_id):
    import torch
    from torch import nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lock = json.loads((ROOT / 'research/hf_lock.json').read_text(encoding='utf-8'))
    entry = lock['models'].get(repo_id)
    if entry is None or not repo_id.startswith('Graf-J/'):
        raise ValueError('Model is not in the reviewed lock file')
    revision = entry['revision']
    local = ROOT / 'models' / repo_id.replace('/', '--')
    config_path = hf_hub_download(repo_id, 'config.json', revision=revision, local_dir=local, token=False)
    processor_path = hf_hub_download(repo_id, 'processor_config.json', revision=revision, local_dir=local, token=False)
    weight_path = hf_hub_download(repo_id, 'model.safetensors', revision=revision, local_dir=local, token=False)
    cfg = json.loads(open(config_path, encoding='utf-8').read())
    processor = json.loads(open(processor_path, encoding='utf-8').read())
    vocab = processor.get('vocab', string.ascii_lowercase + string.ascii_uppercase + string.digits)

    def convolution():
        return nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.BatchNorm2d(32),nn.SiLU(),nn.MaxPool2d(2,2),
                             nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.SiLU(),nn.MaxPool2d(2,2),
                             nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.SiLU(),nn.MaxPool2d((2,1)),
                             nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.SiLU())

    class PositionalEncoding(nn.Module):
        def __init__(self, dim):
            super().__init__()
            pe = torch.zeros(500, dim)
            position = torch.arange(500).unsqueeze(1)
            div = torch.exp(torch.arange(0,dim,2)*(-math.log(10000.0)/dim))
            pe[:,0::2], pe[:,1::2] = torch.sin(position*div), torch.cos(position*div)
            self.register_buffer('pe', pe.unsqueeze(0))
        def forward(self, x):
            return x + self.pe[:,:x.size(1)]

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_crnn = cfg['model_type'] == 'captcha_crnn'
            if self.is_crnn:
                self.conv_layer = convolution()
                self.lstm = nn.LSTM(1280,256,bidirectional=True,batch_first=True)
                self.classifier = nn.Linear(512,cfg['num_chars'])
            else:
                self.conv = convolution()
                self.positional_encoding = PositionalEncoding(cfg['d_model'])
                layer = nn.TransformerEncoderLayer(d_model=cfg['d_model'],nhead=cfg['nhead'],
                    dim_feedforward=cfg['dim_feedforward'],dropout=cfg['dropout'],
                    activation='gelu',batch_first=True,norm_first=True)
                self.transformer = nn.TransformerEncoder(layer, num_layers=cfg['num_layers'],enable_nested_tensor=False)
                self.classifier = nn.Linear(cfg['d_model'],cfg['num_chars'])
        def forward(self, x):
            x = (self.conv_layer if self.is_crnn else self.conv)(x).permute(0,3,1,2)
            x = x.reshape(x.size(0),x.size(1),-1)
            x = self.lstm(x)[0] if self.is_crnn else self.transformer(self.positional_encoding(x))
            return self.classifier(x)

    torch.set_num_threads(2)
    model = Model().eval()
    # Strict matching prevents silent changes in upstream architecture/config.
    model.load_state_dict(load_file(weight_path, device='cpu'), strict=True)
    def predict(image):
        pixels = np.asarray(image.convert('L').resize((150,40)),dtype=np.float32)/255.0
        tensor = torch.from_numpy(pixels.copy())[None,None]
        with torch.inference_mode():
            probs = model(tensor).softmax(-1)[0]
        tokens = probs.argmax(-1).tolist()
        text, scores, previous = [], [], None
        for t, token in enumerate(tokens):
            if token != 0 and token != previous:
                text.append(vocab[token-1])
                scores.append(float(probs[t,token]))
            previous = token
        return ''.join(text), float(np.mean(scores)) if scores else None
    predict.revision = revision
    predict.parameters = sum(p.numel() for p in model.parameters())
    return predict


def load_trocr(repo_id='anuashok/ocr-captcha-v3'):
    """Optional heavier comparison; downloading weights can exceed 1 GB."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    lock = json.loads((ROOT / 'research/hf_lock.json').read_text(encoding='utf-8'))
    revision = lock['models'][repo_id]['revision']
    processor = TrOCRProcessor.from_pretrained(repo_id,revision=revision,trust_remote_code=False,token=False)
    model = VisionEncoderDecoderModel.from_pretrained(repo_id,revision=revision,use_safetensors=True,
                                                     trust_remote_code=False,token=False).eval()
    def predict(image):
        values = processor(images=image.convert('RGB'),return_tensors='pt').pixel_values
        with torch.inference_mode():
            ids = model.generate(values,max_new_tokens=24,num_beams=1)
        return processor.batch_decode(ids,skip_special_tokens=True)[0].strip(), None
    return predict
