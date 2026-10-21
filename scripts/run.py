"""One model, one notebook, one entry point. No paid API calls."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def review():
    from captcha_lab.evaluation import RESULTS,summary_frame
    summary=summary_frame();summary.to_csv(RESULTS/'summary.csv',index=False)
    print(summary[['dataset','n','correct','exact_match','ci95_low','ci95_high','median_ms']].to_string(index=False))


def benchmark():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from captcha_lab.data import save_json
    class CurrentPythonKernel(KernelManager):
        def format_kernel_cmd(self,extra_arguments=None):
            return [sys.executable,'-m','ipykernel_launcher','-f',self.connection_file]+list(extra_arguments or [])
    path=ROOT/'notebooks/final_ppocr_small.ipynb';notebook=nbformat.read(path,4)
    client=NotebookClient(notebook,timeout=600,kernel_name='python3',kernel_manager_class=CurrentPythonKernel,
                          resources={'metadata':{'path':str(ROOT)}})
    result=dict(notebook=path.name,status='running')
    try:
        client.execute();result.update(status='passed',code_cells=sum(c.cell_type=='code' for c in notebook.cells))
    except Exception as exc:
        result['status']=type(exc).__name__;raise
    finally:
        nbformat.write(notebook,path)
        save_json(ROOT/'research/results/notebook_validation.json',result)
    print('PASSED',path.name);review()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['review','benchmark','predict'],nargs='?',default='review')
    parser.add_argument('image',nargs='?',help='Cropped image for predict')
    args=parser.parse_args()
    if args.command=='review':review()
    elif args.command=='benchmark':benchmark()
    else:
        if not args.image:parser.error('predict requires an image path')
        from PIL import Image
        from captcha_lab.model import load_model
        predict=load_model()
        with Image.open(args.image) as image:text,score=predict(image)
        print(text)
