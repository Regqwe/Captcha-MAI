"""Single user-authorized pilot: <=10 attempts, <=$1; no generic API unlock.

The ledger is persisted BEFORE each request. Failed/uncertain requests consume a
slot and keep their full reservation. Completed pilots cannot be restarted.
"""
import base64
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import random
import re
import time

import pandas as pd
from PIL import Image
import requests
from requests.adapters import HTTPAdapter

from .data import ROOT
from .gemini import GeminiClient

FOLDER=ROOT/'research/results_v3'
MODELS=['gemini-3.1-flash-lite','gemini-3.5-flash-lite']
PRICES={MODELS[0]:('0.25','1.50'),MODELS[1]:('0.30','2.50')}
PROMPT=('Transcribe the characters in this image in order. Return only the string, '
        'preserving case and leading zeros. If unreadable, return UNREADABLE.')
CONFIG=dict(temperature=0,candidateCount=1,maxOutputTokens=512,
            thinkingConfig=dict(thinkingLevel='MINIMAL'))
MAX_ATTEMPTS=10
MAX_USD=Decimal('1.00')
RESERVE_USD=Decimal('0.01')  # >6x worst configured token cost; never released.
INPUT_TOKEN_BOUND=1024  # one <=384px image (258 documented tokens) + short ASCII prompt + margin


def now():return datetime.now(timezone.utc).isoformat()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare():
    """Deterministic, label-independent selection; no network."""
    FOLDER.mkdir(exist_ok=True)
    source=pd.read_csv(ROOT/'datasets/target/annotations.csv',dtype=str,keep_default_na=False)
    eligible=source[source.label.ne('') & source.label_status.ne('ambiguous')]
    ids=sorted(random.Random(20261022).sample(sorted(eligible.image_id),5))
    selected=eligible.set_index('image_id').loc[ids].reset_index().copy()
    inputs=FOLDER/'inputs';inputs.mkdir(exist_ok=True)
    records=[]
    for row in selected.itertuples():
        with Image.open(ROOT/row.image_path) as im:
            image=im.convert('RGB');image.thumbnail((384,384),Image.Resampling.LANCZOS)
        stream=io.BytesIO();image.save(stream,format='PNG');raw=stream.getvalue()
        path=inputs/(row.image_id+'.png');path.write_bytes(raw)
        records.append(dict(image_id=row.image_id,path=str(path.relative_to(ROOT)).replace('\\','/'),
            sha256=hashlib.sha256(raw).hexdigest(),width=image.width,height=image.height))
    selected.to_csv(FOLDER/'selected_labels.csv',index=False)
    bounds={m:str((Decimal(INPUT_TOKEN_BOUND)*Decimal(PRICES[m][0])+
                  Decimal(CONFIG['maxOutputTokens'])*Decimal(PRICES[m][1]))/Decimal(1_000_000)) for m in MODELS}
    plan=dict(authorization='User: at most 10 measurements, at most USD 1; 2026-10-21',
        attempt_limit=MAX_ATTEMPTS,usd_limit=str(MAX_USD),reservation_per_attempt_usd=str(RESERVE_USD),
        planned_reservation_usd=str(MAX_ATTEMPTS*RESERVE_USD),models=MODELS,
        images=records,prompt=PROMPT,generation_config=CONFIG,input_token_bound=INPUT_TOKEN_BOUND,
        price_usd_per_million=PRICES,per_request_configured_cost_bound_usd=bounds,
        configured_total_cost_bound_usd=str(sum(Decimal(v)*5 for v in bounds.values())),
        seed=20261022,retries=0,workers=1,selection='five of seven visually labeled development targets, identical images per model',
        task='offline transcription; no live service submission',
        sources=['https://ai.google.dev/gemini-api/docs/pricing',
            'https://ai.google.dev/gemini-api/docs/image-understanding',
            'https://ai.google.dev/gemini-api/docs/generate-content/thinking.md'])
    path=FOLDER/'plan.json'
    if path.exists() and fingerprint(json.loads(path.read_text(encoding='utf-8')))!=fingerprint(plan):
        raise RuntimeError('Frozen pilot plan changed; refusing to spend')
    atomic_json(path,plan)
    return plan


class BudgetStop(RuntimeError):pass


class Ledger:
    def __init__(self,path,plan_hash):
        self.path=Path(path)
        if self.path.exists():
            self.data=json.loads(self.path.read_text(encoding='utf-8'))
            if self.data['plan_sha256']!=plan_hash:raise BudgetStop('Plan mismatch')
        else:
            self.data=dict(plan_sha256=plan_hash,attempt_limit=MAX_ATTEMPTS,
                usd_limit=str(MAX_USD),closed=False,created_at=now(),attempts=[])
            self.save()

    def save(self):atomic_json(self.path,self.data)

    def reserve(self,model,image_id):
        attempts=self.data['attempts']
        if self.data['closed']:raise BudgetStop('Pilot closed; new user authorization required')
        if len(attempts)>=MAX_ATTEMPTS:raise BudgetStop('Ten-attempt limit exhausted')
        if any(a['model']==model and a['image_id']==image_id for a in attempts):
            raise BudgetStop('Attempt already reserved; retries forbidden')
        reserved=sum(Decimal(a['reserved_usd']) for a in attempts)
        if reserved+RESERVE_USD>MAX_USD:raise BudgetStop('Dollar limit would be exceeded')
        attempt=dict(attempt=len(attempts)+1,model=model,image_id=image_id,
                     reserved_usd=str(RESERVE_USD),started_at=now(),status='reserved_before_http')
        attempts.append(attempt);self.save()
        return attempt

    def finish(self,attempt,result):
        attempt.update(result);attempt['finished_at']=now();self.save()

    def close(self,reason):
        self.data.update(closed=True,close_reason=reason,closed_at=now());self.save()


def execute():
    """Only called explicitly by the one-off runner; ledger survives restarts."""
    raise BudgetStop('Gemini pilot cancelled by the user. No network request was sent.')
    plan=prepare()
    lock=FOLDER/'pilot.lock'
    with lock.open('x',encoding='utf-8') as handle:handle.write(now())
    session=None
    try:
        ledger=Ledger(FOLDER/'budget_ledger.json',fingerprint(plan))
        if ledger.data['closed']:raise BudgetStop('Authorized pilot already closed; no calls allowed')
        credentials=GeminiClient.from_env()  # generic client's paid methods stay disabled
        session=requests.Session();session.trust_env=False
        session.proxies={'https':credentials.proxy_url,'http':credentials.proxy_url}
        session.mount('https://',HTTPAdapter(max_retries=0))
        disabled_models=set();stop_reason='planned_attempts_completed'
        for item in plan['images']:
            raw=(ROOT/item['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise BudgetStop('Input image changed')
            assert item['width']<=384 and item['height']<=384
            assert len(PROMPT.encode('ascii'))<256 and len(raw)<200000
            for model in MODELS:
                if model in disabled_models:continue
                if any(a['model']==model and a['image_id']==item['image_id'] for a in ledger.data['attempts']):continue
                payload=dict(contents=[dict(parts=[dict(text=PROMPT),dict(inlineData=dict(
                    mimeType='image/png',data=base64.b64encode(raw).decode('ascii')))])],generationConfig=CONFIG)
                attempt=ledger.reserve(model,item['image_id'])  # fsync boundary via atomic replacement BEFORE send
                result=dict(prediction='',status='error',error='',elapsed_ms=0,model_version='',
                            prompt_tokens=None,candidate_tokens=None,thinking_tokens=None,total_tokens=None,
                            estimated_usage_cost_usd=None,finish_reason='',http_status=None)
                start=time.perf_counter();abort=False
                try:
                    response=session.post('https://generativelanguage.googleapis.com/v1beta/models/'+model+':generateContent',
                        headers={'X-goog-api-key':credentials.api_key},json=payload,timeout=(15,60),allow_redirects=False)
                    result['http_status']=response.status_code
                    if response.status_code!=200:
                        result['error']='HTTP_'+str(response.status_code)
                        disabled_models.add(model)
                    else:
                        body=response.json();usage=body.get('usageMetadata',{})
                        candidates=body.get('candidates',[]);candidate=candidates[0] if candidates else {}
                        result.update(model_version=body.get('modelVersion',model),
                            finish_reason=candidate.get('finishReason','NO_CANDIDATE'),
                            prompt_tokens=usage.get('promptTokenCount'),candidate_tokens=usage.get('candidatesTokenCount',0),
                            thinking_tokens=usage.get('thoughtsTokenCount',0),total_tokens=usage.get('totalTokenCount'))
                        result['prediction']=''.join(p.get('text','') for p in candidate.get('content',{}).get('parts',[]) if not p.get('thought')).strip()
                        if result['prompt_tokens'] is not None and result['total_tokens'] is not None:
                            prompt=int(result['prompt_tokens'])
                            # Conservative when provider omits a breakdown: include every non-input token.
                            output=max(int(result['candidate_tokens'])+int(result['thinking_tokens']),int(result['total_tokens'])-prompt)
                            cost=(Decimal(prompt)*Decimal(PRICES[model][0])+Decimal(output)*Decimal(PRICES[model][1]))/Decimal(1_000_000)
                            result['estimated_usage_cost_usd']=str(cost)
                            if prompt>INPUT_TOKEN_BOUND or output>CONFIG['maxOutputTokens'] or cost>RESERVE_USD:
                                abort=True;result['error']='PROVIDER_TOKEN_BOUND_VIOLATION'
                        else:
                            abort=True;result['error']='MISSING_USAGE_STOP'
                        if not result['error']:
                            if result['finish_reason']!='STOP':result['error']='FINISH_'+result['finish_reason']
                            elif not re.fullmatch(r'[A-Za-z0-9]{1,32}',result['prediction']) or result['prediction']=='UNREADABLE':
                                result['error']='INVALID_FORMAT_OR_ABSTENTION'
                            else:result['status']='ok'
                except requests.RequestException:
                    result['error']='NETWORK_FAILURE_NO_RETRY';abort=True
                except (ValueError,TypeError):
                    result['error']='RESPONSE_PARSE_FAILURE_NO_RETRY';abort=True
                result['elapsed_ms']=(time.perf_counter()-start)*1000
                ledger.finish(attempt,result)
                print('Attempt',attempt['attempt'],'/10',model,item['image_id'],result['status'],result['error'],flush=True)
                if abort:
                    stop_reason=result['error'];ledger.close(stop_reason);return ledger.data
        ledger.close(stop_reason)
        return ledger.data
    finally:
        if session is not None:session.close()
        lock.unlink(missing_ok=True)


def export_rows():
    data=json.loads((FOLDER/'budget_ledger.json').read_text(encoding='utf-8'))
    frame=pd.DataFrame(data['attempts'])
    labels=pd.read_csv(FOLDER/'selected_labels.csv',dtype=str,keep_default_na=False)
    frame=frame.merge(labels[['image_id','label']],on='image_id',validate='many_to_one')
    frame['correct']=frame.status.eq('ok') & frame.prediction.eq(frame.label)
    frame.to_csv(FOLDER/'gemini_attempts.csv',index=False)
    return frame
