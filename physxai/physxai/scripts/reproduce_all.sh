#!/bin/bash
echo "==> Setting up environment"
conda env create -f environment.yml
conda activate physxai
pip install -e .

echo "==> Downloading data"
python scripts/download_data.py

echo "==> Preprocessing data"
python scripts/preprocess_data.py

echo "==> Generating synthetic data"
python scripts/generate_synthetic.py

echo "==> Training model"
python scripts/train.py

echo "==> Evaluating model"
python scripts/evaluate.py

echo "==> Generating figures"
python scripts/generate_figures.py

echo "==> Done!"