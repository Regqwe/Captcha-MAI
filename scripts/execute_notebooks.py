"""Run notebooks using this interpreter's kernel, saving outputs only locally."""
import argparse
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern',default='*.ipynb')
    parser.add_argument('--skip-api',action='store_true')
    args=parser.parse_args()
    target=ROOT/'notebooks/executed';target.mkdir(exist_ok=True)
    statuses=[]
    for path in sorted((ROOT/'notebooks').glob(args.pattern)):
        if args.skip_api and path.name.startswith('03_'):
            continue
        print('EXECUTE',path.name,flush=True)
        notebook=nbformat.read(path,as_version=4)
        for cell in notebook.cells:
            if cell.cell_type=='code': compile(cell.source,str(path),'exec')
        client=NotebookClient(notebook,timeout=600,kernel_name='captcha-mai',
                              resources={'metadata':{'path':str(ROOT)}})
        status='passed'
        try:
            client.execute()
        except Exception as exc:
            status=type(exc).__name__
            print('FAILED',path.name,status,flush=True)
        nbformat.write(notebook,target/path.name)
        statuses.append({'notebook':path.name,'status':status})
        print('RESULT',path.name,status,flush=True)
    result=ROOT/'results/notebook_validation.json'
    result.parent.mkdir(exist_ok=True)
    previous = json.loads(result.read_text(encoding='utf-8')) if result.exists() else []
    merged = {r['notebook']:r for r in previous}
    merged.update({r['notebook']:r for r in statuses})
    result.write_text(json.dumps(list(merged.values()),indent=2),encoding='utf-8')
    if any(r['status']!='passed' for r in statuses):raise SystemExit(1)


if __name__=='__main__':main()
