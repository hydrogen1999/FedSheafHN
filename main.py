# main.py

import argparse
import yaml
import numpy as np

from fl.main import fl_main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n_runs", type=int, default=1)
    args_cli = parser.parse_args()

    with open(args_cli.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Kiểm tra mode
    if cfg.get('mode')=='disjoint':
        print("==> Generating disjoint partition data...")
        generate_disjoint_data(cfg)
    else:
        print(f"mode={cfg.get('mode')} => skip or use other generator?")

    # Sau đó chạy FL pipeline
    fl_main(cfg)


if __name__=="__main__":
    main()
