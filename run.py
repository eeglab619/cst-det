import argparse
import json
from pathlib import Path
from release import RELEASE_NOTICE, require_core

def main():
    parser = argparse.ArgumentParser(description=RELEASE_NOTICE)
    parser.add_argument('--describe', action='store_true', help='Show protocol; no numerical experiment')
    parser.add_argument('--data', type=Path, help='Prepared single-participant NPZ cache')
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    if args.describe:
        print(RELEASE_NOTICE)
        print(json.dumps(config, indent=2))
        return
    require_core()
    if args.data is None:
        parser.error('--data is required')
    from pipeline import run_experiment
    print(run_experiment(args.data, config))
if __name__ == '__main__':
    main()
