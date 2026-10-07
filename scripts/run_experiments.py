"""从项目根目录运行 03、04，并保存输出。"""
from pathlib import Path
import sys
import argparse
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
from experiment_validation import capture_environment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', choices=['03','04','both'], default='both')
    args = parser.parse_args()
    capture_environment(root)
    names = ['03_feature_and_model_comparison.ipynb','04_results_and_conclusions.ipynb']
    for name in names:
        if args.step != 'both' and not name.startswith(args.step): continue
        path = root / 'experiment_steps' / name
        experiment = nbformat.read(path, as_version=4)
        def progress(cell, cell_index, **kwargs):
            if cell.cell_type == 'code':
                print(f'{name} | {cell.source.splitlines()[0]}', flush=True)
        client = NotebookClient(experiment, timeout=7200, kernel_name='python3',
                                resources={'metadata':{'path':str(root)}}, on_cell_start=progress)
        try:
            client.execute()
        finally:
            nbformat.write(experiment, path)
        print(f'已完成并保存：{name}', flush=True)

if __name__ == '__main__': main()
