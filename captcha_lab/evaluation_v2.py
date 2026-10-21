"""Real inference on the frozen labeled holdout, never mocked predictions."""
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
from statistics import NormalDist
import time

import numpy as np
import pandas as pd
from PIL import Image
import psutil

from .benchmark import edit_distance
from .data import ROOT, save_json

RESULTS = ROOT / 'results/v2'
PROMPT = ('Transcribe only the characters in this image, in order. Return just the string, '
          'preserving case and leading zeros. If unreadable, return exactly UNREADABLE.')
LOCAL_MODELS = ['ppocr-tiny','ppocr-small','ddddocr-beta','crnn-base','conv-transformer-base']


def manifest(name='holdout'):
    if name == 'target':
        path = ROOT / 'datasets/target/annotations.csv'
    else:
        path = ROOT / 'datasets/captcha-images' / (name+'_labels.csv')
    frame = pd.read_csv(path,dtype=str,keep_default_na=False)
    if not frame.image_id.is_unique:
        raise ValueError('Duplicate image IDs')
    frame['path'] = frame.image_path.map(lambda p:str(ROOT/p))
    if not frame.path.map(lambda p:Path(p).is_file()).all():
        raise FileNotFoundError('Run scripts/prepare_benchmark.py to extract the real dataset')
    return frame


def factory(model):
    from .onnx_models import load_ddddocr, load_ppocr
    if model == 'ppocr-tiny':return load_ppocr('tiny')
    if model == 'ppocr-small':return load_ppocr('small')
    if model == 'ddddocr-beta':return load_ddddocr(True)
    from .hf_models import load_graf
    if model == 'crnn-base':return load_graf('Graf-J/captcha-crnn-base')
    if model == 'conv-transformer-base':return load_graf('Graf-J/captcha-conv-transformer-base')
    raise ValueError('Unknown frozen model')


def _local_row(item, predictor, model, dataset):
    start=time.perf_counter()
    prediction, score, error, detail='',None,'',''
    try:
        with Image.open(item.path) as im:image=im.convert('RGB')
        prediction,score=predictor(image)
    except Exception as exc:
        error=type(exc).__name__
        detail=getattr(exc,'safe_message','')
    elapsed=(time.perf_counter()-start)*1000
    return dict(model=model,dataset=dataset,image_id=item.image_id,prediction=str(prediction),
                score=score,elapsed_ms=elapsed,error=error,error_detail=detail)


def run_local(model):
    """Always performs model inference; does not load old prediction CSVs."""
    RESULTS.mkdir(parents=True,exist_ok=True)
    folder=RESULTS/model;folder.mkdir(exist_ok=True)
    start=time.perf_counter();predictor=factory(model);init_ms=(time.perf_counter()-start)*1000
    validation=manifest('validation')
    warm=_local_row(next(validation.itertuples()),predictor,model,'warmup')
    if warm['error']:raise RuntimeError('Warmup failed: '+warm['error'])
    frames={}
    max_rss=psutil.Process().memory_info().rss
    for dataset in ['holdout','target']:
        data=manifest(dataset);rows=[]
        for i,item in enumerate(data.itertuples(),1):
            rows.append(_local_row(item,predictor,model,dataset))
            max_rss=max(max_rss,psutil.Process().memory_info().rss)
            if i%200==0:print(model,dataset,i,'/',len(data),flush=True)
        frame=pd.DataFrame(rows);frame.to_csv(folder/(dataset+'.csv'),index=False)
        frames[dataset]=frame
    versions={}
    for p in ['numpy','pillow','onnxruntime','rapidocr','ddddocr','torch']:
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:pass
    save_json(folder/'run.json',dict(model=model,init_ms=init_ms,warmup_ms=warm['elapsed_ms'],
        observed_rss_max_mb=max_rss/1024**2,rss_scope='whole kernel, sampled after each image; not isolated weight memory',
        platform=platform.platform(),versions=versions,parameters=getattr(predictor,'parameters',None),
        revision=getattr(predictor,'revision',None),batch_size=1,cache_predictions=False))
    return frames


def run_gemini():
    raise RuntimeError('Paid experiment disabled. Use saved results; review model and budget before any new API experiment.')


def wilson(correct,total,confidence=.95):
    if total==0:return (None,None)
    z=NormalDist().inv_cdf((1+confidence)/2)
    p=correct/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return max(0.,center-radius),min(1.,center+radius)


def scored(predictions,labels):
    labels=labels[labels.label.ne('') & labels.label_status.ne('ambiguous')]
    if predictions.image_id.duplicated().any() or labels.image_id.duplicated().any():
        raise ValueError('Duplicate IDs would inflate accuracy')
    joined=labels[['image_id','label']].merge(predictions,on='image_id',how='left',validate='one_to_one')
    joined['error']=joined.error.fillna('MissingPrediction')
    joined['prediction']=joined.prediction.fillna('').astype(str)
    joined['correct']=joined.error.eq('') & joined.prediction.eq(joined.label)
    joined['edits']=[edit_distance(p if not err else '',label) for p,err,label in
                     zip(joined.prediction,joined.error,joined.label)]
    return joined


def metrics(predictions,labels):
    frame=scored(predictions,labels);n=len(frame);k=int(frame.correct.sum())
    lo,hi=wilson(k,n)
    elapsed=pd.to_numeric(frame.elapsed_ms,errors='coerce')
    successful=elapsed[frame.error.eq('')]
    return dict(n=n,correct=k,exact_match=k/n if n else None,ci95_low=lo,ci95_high=hi,
                cer=frame.edits.sum()/frame.label.str.len().sum() if n else None,
                errors=int(frame.error.ne('').sum()),missing=int(frame.error.eq('MissingPrediction').sum()),
                median_ms=elapsed.median(),p95_ms=elapsed.quantile(.95),
                success_median_ms=successful.median(),success_p95_ms=successful.quantile(.95))


def load_predictions(model,dataset):
    path=RESULTS/model/(dataset+'.csv')
    if not path.exists():
        path=ROOT/'research/results_v2'/model/(dataset+'.csv')
    return pd.read_csv(path,dtype={'image_id':str,'prediction':str},keep_default_na=False)


def compare(model_a,model_b,labels,dataset_a='holdout',dataset_b='holdout'):
    a=scored(load_predictions(model_a,dataset_a),labels).set_index('image_id').correct
    b=scored(load_predictions(model_b,dataset_b),labels).set_index('image_id').correct.reindex(a.index)
    wins=int((a & ~b).sum());losses=int((~a & b).sum());n=len(a)
    # Bonferroni across wins/losses makes the difference interval conservative.
    wl,wu=wilson(wins,n,.975);ll,lu=wilson(losses,n,.975)
    discordant=wins+losses
    if discordant:
        k=min(wins,losses)
        logs=[math.lgamma(discordant+1)-math.lgamma(j+1)-math.lgamma(discordant-j+1)-discordant*math.log(2)
              for j in range(k+1)]
        peak=max(logs)
        p=min(1.,2*math.exp(peak)*sum(math.exp(x-peak) for x in logs))
    else:p=1.
    return dict(model_a=model_a,model_b=model_b,n=n,wins=wins,losses=losses,
                delta=(wins-losses)/n,delta_ci95_low=wl-lu,delta_ci95_high=wu-ll,mcnemar_exact_p=p)


def all_summaries():
    rows=[]
    for model in LOCAL_MODELS+['gemini']:
        for dataset in ['holdout','shared_200','target']:
            if model=='gemini' and dataset=='holdout':continue
            try:
                pred=load_predictions(model,'holdout' if dataset=='shared_200' and model!='gemini' else dataset)
            except FileNotFoundError:continue
            labels=manifest(dataset)
            result=metrics(pred,labels)
            rows.append(dict(model=model,dataset=dataset,**result))
    return pd.DataFrame(rows)


def hypothesis_results():
    summary=all_summaries()
    def row(model,dataset='holdout'):
        return summary[(summary.model==model)&(summary.dataset==dataset)].iloc[0]
    output=[]
    h1=row('ppocr-tiny')
    verdict='supported' if h1.ci95_low>=.95 and h1.p95_ms<=100 else (
        'rejected' if h1.ci95_high<.95 or h1.p95_ms>100 else 'inconclusive')
    output.append(dict(hypothesis='H1',criterion='tiny exact>=95%, p95<=100ms',verdict=verdict,
                       estimate=h1.exact_match,ci_low=h1.ci95_low,ci_high=h1.ci95_high))
    configs=[('H2','ppocr-small','ppocr-tiny',.02,'holdout',True),
             ('H3','ddddocr-beta','ppocr-small',0.,'holdout',False),
             ('H4','crnn-base','ppocr-small',-.02,'holdout',False),
             ('H5','conv-transformer-base','crnn-base',.02,'holdout',False),
             ('H6','gemini','ppocr-small',-.02,'shared_200',False)]
    paired=[]
    for h,a,b,threshold,ds,latency in configs:
        try:
            result=compare(a,b,manifest(ds),ds,'holdout')
        except FileNotFoundError:continue
        good=result['delta_ci95_low']>=threshold
        bad=result['delta_ci95_high']<threshold
        if latency and row(a).p95_ms>150:bad=True;good=False
        verdict='supported' if good else 'rejected' if bad else 'inconclusive'
        paired.append(dict(hypothesis=h,**result))
        output.append(dict(hypothesis=h,criterion=f'delta {a}-{b}>={threshold:.2%}',verdict=verdict,
                           estimate=result['delta'],ci_low=result['delta_ci95_low'],ci_high=result['delta_ci95_high']))
    if paired:
        order=sorted(range(len(paired)),key=lambda i:paired[i]['mcnemar_exact_p']);maximum=0
        for rank,i in enumerate(order):
            maximum=max(maximum,min(1.,(len(paired)-rank)*paired[i]['mcnemar_exact_p']))
            paired[i]['holm_p']=maximum
    return pd.DataFrame(output),pd.DataFrame(paired)
