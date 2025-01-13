# main.py
import yaml
import argparse
import numpy as np
from fl.main import fl_main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n_runs", type=int, default=1)
    args_cli = parser.parse_args()

    # Đọc file YAML
    with open(args_cli.config, 'r') as f:
        config = yaml.safe_load(f)

    n_runs = args_cli.n_runs
    best_test_list = []
    for run_id in range(n_runs):
        # fix seed
        config['seed'] = config['seed'] + run_id
        # chạy FL
        fl_main(config)
        # TODO: Lưu test acc => best_test_list
        # ...
        print(f"Finished run {run_id}")

    if len(best_test_list) > 0:
        print("Best test:", np.mean(best_test_list))

if __name__ == "__main__":
    main()
