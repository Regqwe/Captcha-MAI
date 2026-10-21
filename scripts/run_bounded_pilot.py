"""Explicit one-off runner; default only prepares/prints a frozen plan."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from captcha_lab.bounded_pilot import execute,export_rows,prepare

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-authorized-pilot',action='store_true')
    args=parser.parse_args()
    if args.run_authorized_pilot:
        execute();export_rows()
    else:
        print(json.dumps(prepare(),indent=2))
