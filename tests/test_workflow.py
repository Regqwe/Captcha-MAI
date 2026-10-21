"""Integration checks for saved evidence and the single offline entry point."""
import unittest
import pandas as pd
from captcha_lab.data import ROOT
from captcha_lab.evaluation import LOCAL_MODELS,all_summaries,load_predictions,manifest,metrics


class SavedEvidence(unittest.TestCase):
    def test_all_five_complete_local_runs_are_readable(self):
        labels=manifest('holdout')
        self.assertEqual(len(labels),1984)
        for model in LOCAL_MODELS:
            predictions=load_predictions(model,'holdout')
            self.assertEqual(set(predictions.image_id),set(labels.image_id))
            result=metrics(predictions,labels)
            self.assertEqual(result['missing'],0)
            self.assertEqual(result['errors'],0)

    def test_canonical_csv_contains_labels_and_all_target_attempts(self):
        for model in LOCAL_MODELS+['gemini']:
            raw=pd.read_csv(ROOT/'research/results'/model/'target.csv',dtype=str,keep_default_na=False)
            self.assertEqual(len(raw),9)
            self.assertEqual(int(raw.label.ne('').sum()),7)
            self.assertEqual(int(raw.label_status.eq('ambiguous').sum()),2)

    def test_six_notebooks_use_one_evaluation_module(self):
        import nbformat
        paths=list((ROOT/'notebooks').glob('*.ipynb'))
        self.assertEqual(len(paths),6)
        for path in paths:
            notebook=nbformat.read(path,4)
            code='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
            self.assertIn('captcha_lab.evaluation',code)
            self.assertNotIn('run_gemini',code)
            self.assertNotIn('generateContent',code)


if __name__=='__main__':unittest.main()
