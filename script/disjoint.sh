#!/usr/bin/env bash
set -e

CONFIG="configs/default.yaml"

echo "=== Run pipeline with config: $CONFIG ==="
python main.py --config "$CONFIG"

echo "=== DONE ==="
