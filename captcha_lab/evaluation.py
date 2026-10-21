"""Evaluation of the single final model on real labeled images."""
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from statistics import NormalDist
import time
import pandas as pd
from PIL import Image
import psutil
from .data import ROOT,save_json

MODEL='ppocr-small'
RESULTS=ROOT/'research/results'


def manifest(name='holdout', *, require_images=False):
    if name not in {'holdout','target'}:raise ValueError('Choose holdout or target')
    path=ROOT/('datasets/holdout/labels.csv' if name=='holdout' else 'datasets/target/annotations.csv')
    frame=pd.read_csv(path,dtype=str,keep_default_na=False)
    if not frame.image_id.is_unique:raise ValueError('Duplicate image IDs')
    frame['path']=frame.image_path.map(lambda value:str(ROOT/value))
    if require_images:
        for row in frame.itertuples():
            image=Path(row.path)
            if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest()!=row.sha256:
                raise RuntimeError('Missing or changed evaluation image: '+row.image_id)
    return frame


def load_predictions(dataset):
    return pd.read_csv(RESULTS/(dataset+'.csv'),dtype={'image_id':str,'label':str,'prediction':str},keep_default_na=False)


def run_benchmark():
    from .model import load_model
    for dataset in ['holdout','target']:manifest(dataset,require_images=True)
    started=time.perf_counter();predict=load_model();init_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter();predict(Image.new('RGB',(160,48),'white'))
    warmup_ms=(time.perf_counter()-started)*1000
    frames={};max_rss=psutil.Process().memory_info().rss
    for dataset in ['holdout','target']:
        labels=manifest(dataset);rows=[]
        for i,row in enumerate(labels.itertuples(),1):
            start=time.perf_counter();prediction='';score=None;error=''
            try:
                with Image.open(row.path) as image:prediction,score=predict(image.convert('RGB'))
            except Exception as exc:error=type(exc).__name__
            rows.append(dict(image_id=row.image_id,model=MODEL,dataset=dataset,prediction=str(prediction),
                             score=score,error=error,elapsed_ms=(time.perf_counter()-start)*1000))
            max_rss=max(max_rss,psutil.Process().memory_info().rss)
            if i%200==0:print(MODEL,dataset,i,'/',len(labels),flush=True)
        predictions=pd.DataFrame(rows)
        frame=labels[['image_id','label','label_status']].merge(predictions,on='image_id',validate='one_to_one')
        eligible=frame.label.ne('') & frame.label_status.ne('ambiguous')
        frame['correct']=pd.array([None]*len(frame),dtype='boolean')
        frame.loc[eligible,'correct']=frame.loc[eligible,'error'].eq('') & frame.loc[eligible,'prediction'].eq(frame.loc[eligible,'label'])
        RESULTS.mkdir(exist_ok=True);frame.to_csv(RESULTS/(dataset+'.csv'),index=False)
        frames[dataset]=frame
    save_json(RESULTS/'run.json',dict(model=MODEL,init_ms=init_ms,warmup_ms=warmup_ms,
        warmup='blank white image, excluded from evaluation',batch_size=1,cache_predictions=False,
        observed_rss_max_mb=max_rss/1024**2,rss_scope='whole kernel sampled after each image',
        platform=platform.platform(),versions={name:importlib.metadata.version(name) for name in
            ['numpy','pillow','pandas','rapidocr','onnxruntime']},
        weight_sha256=json.loads((ROOT/'research/model.json').read_text())['sha256'],
        holdout_manifest_sha256=hashlib.sha256((ROOT/'datasets/holdout/labels.csv').read_bytes()).hexdigest(),
        timing_scope='image open + preprocessing + inference + decoding; initialization/warmup separate'))
    return frames


def summary_frame():
    return pd.DataFrame([dict(model=MODEL,dataset=dataset,**metrics(load_predictions(dataset),manifest(dataset)))
                         for dataset in ['holdout','target']])


def wilson(correct,total,confidence=.95):
    if total==0:return (None,None)
    z=NormalDist().inv_cdf((1+confidence)/2)
    p=correct/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return max(0.,center-radius),min(1.,center+radius)


def scored(predictions,labels):
    predictions=predictions.drop(columns=['label','label_status','correct'],errors='ignore')
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


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return previous[-1]
