"""One local model: evaluation, image recognition and Android screen watcher."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command')
    sub.add_parser('review',help='View saved real evaluation')
    sub.add_parser('benchmark',help='Run the final notebook with actual inference')
    sub.add_parser('phone-eval',help='Actual inference on labeled Drive examples')
    pred=sub.add_parser('predict',help='Recognize an already cropped image')
    pred.add_argument('image',type=Path)
    check=sub.add_parser('phone-check',help='List ADB devices and authorization states')
    cap=sub.add_parser('phone-capture',help='Save a phone screenshot and UI XML locally')
    cap.add_argument('--serial')
    cap.add_argument('--output',type=Path,default=ROOT/'runs/capture')
    phone=sub.add_parser('phone',help='Watch and submit supported challenges; Ctrl+C stops')
    phone.add_argument('--serial')
    phone.add_argument('--profile',type=Path,default=ROOT/'phone-profile.json')
    phone.add_argument('--observe',action='store_true',help='Recognize without typing or clicking')
    phone.add_argument('--duration',type=float,default=0,help='Seconds, 0 means until Ctrl+C')
    phone.add_argument('--output',type=Path,help='Local event log folder (default runs/timestamp)')
    args=parser.parse_args()
    if args.command in (None,'review'):review()
    elif args.command=='benchmark':benchmark()
    elif args.command=='predict':
        from PIL import Image
        from captcha_lab.model import load_model
        predict=load_model()
        with Image.open(args.image) as image:text,score=predict(image)
        print(text)
    elif args.command=='phone-check':
        import json
        from captcha_lab.android import device_inventory
        devices=device_inventory()
        print(json.dumps(devices,ensure_ascii=False,indent=2))
        if not devices:print('Телефон не найден. Подключите USB и разрешите отладку на телефоне.')
    elif args.command=='phone-capture':
        from captcha_lab.android import capture
        capture(args.serial,args.output)
    elif args.command=='phone-eval':
        from captcha_lab.android import evaluate_phone
        evaluate_phone()
    else:
        from captcha_lab.android import watch
        if args.duration<0:parser.error('--duration must be >= 0')
        watch(args.serial,args.profile,observe=args.observe,duration=args.duration,output=args.output)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
