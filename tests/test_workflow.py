import unittest
from pathlib import Path
import nbformat
from captcha_lab.data import ROOT
from captcha_lab.evaluation import manifest,load_predictions,metrics


class FinalWorkflow(unittest.TestCase):
    def test_complete_real_holdout_with_ground_truth(self):
        labels=manifest('holdout',require_images=True)
        predictions=load_predictions('holdout')
        self.assertEqual(len(labels),1984)
        self.assertEqual(set(labels.image_id),set(predictions.image_id))
        result=metrics(predictions,labels)
        self.assertEqual(result['correct'],1876)
        self.assertEqual(result['missing'],0)
        self.assertEqual(result['errors'],0)

    def test_all_nine_targets_preserve_ambiguous_labels(self):
        frame=load_predictions('target')
        self.assertEqual(len(frame),9)
        self.assertEqual(int(frame.label.ne('').sum()),7)
        self.assertEqual(int(frame.label_status.eq('ambiguous').sum()),2)

    def test_one_notebook_and_no_model_or_archive_extensions(self):
        self.assertEqual([p.name for p in (ROOT/'notebooks').glob('*.ipynb')],['final_ppocr_small.ipynb'])
        self.assertFalse((ROOT/'captcha_lab/hf_models.py').exists())
        self.assertFalse((ROOT/'captcha_lab/gemini.py').exists())
        self.assertFalse(list((ROOT/'datasets').rglob('*.gz')))
        n=nbformat.read(ROOT/'notebooks/final_ppocr_small.ipynb',4)
        code='\n'.join(c.source for c in n.cells if c.cell_type=='code')
        self.assertIn('run_benchmark()',code)
        self.assertNotIn('generateContent',code)


if __name__=='__main__':unittest.main()
