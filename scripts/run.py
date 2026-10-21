"""Single offline entry point: prepare data, review saved results, or rerun all notebooks."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def review():
    from captcha_lab.evaluation import RESULTS,all_summaries,hypothesis_results
    summary=all_summaries();hypotheses,paired=hypothesis_results()
    summary.to_csv(RESULTS/'summary.csv',index=False)
    hypotheses.to_csv(RESULTS/'hypotheses.csv',index=False)
    paired.to_csv(RESULTS/'paired_comparisons.csv',index=False)
    print(summary[['model','dataset','n','correct','exact_match','median_ms']].to_string(index=False))
    print(hypotheses[['hypothesis','verdict']].to_string(index=False))


def benchmark():
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from captcha_lab.dataset import prepare
    from captcha_lab.data import save_json
    prepare()
    notebooks=sorted((ROOT/'notebooks').glob('[1-6]_*.ipynb'))
    if len(notebooks)!=6:raise RuntimeError('Expected exactly six canonical notebooks')
    records=[]
    class CurrentPythonKernel(KernelManager):
        def format_kernel_cmd(self,extra_arguments=None):
            return [sys.executable,'-m','ipykernel_launcher','-f',self.connection_file]+list(extra_arguments or [])
    for path in notebooks:
        print('EXECUTE',path.name,flush=True)
        notebook=nbformat.read(path,as_version=4)
        # The current interpreter is the only kernel; no preinstalled named kernel is required.
        client=NotebookClient(notebook,timeout=600,kernel_name='python3',
            kernel_manager_class=CurrentPythonKernel,
            resources={'metadata':{'path':str(ROOT)}})
        record=dict(notebook=path.name,status='running')
        try:
            client.execute()
            record.update(status='passed',code_cells=sum(c.cell_type=='code' for c in notebook.cells))
        except Exception as exc:
            record.update(status=type(exc).__name__)
            raise
        finally:
            nbformat.write(notebook,path)
            records.append(record)
            save_json(ROOT/'research/results/notebook_validation.json',records)
        print('PASSED',path.name,flush=True)
    review()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['review','prepare','benchmark'],nargs='?',default='review')
    args=parser.parse_args()
    if args.command=='review':review()
    elif args.command=='prepare':
        from captcha_lab.dataset import prepare
        prepare()
    else:benchmark()
