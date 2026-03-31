#!/bin/bash

set -e

echo "Starting Traffic Sign Classifier Pipeline"

echo -e "\n[1/6] Setting up Python Virtual Environment..."
if [ ! -d "venv" ]; then
    python3.12 -m venv venv
fi

source venv/Scripts/activate

echo "Installing dependencies (TensorFlow, Seaborn, etc.)..."
pip install numpy matplotlib seaborn tensorflow scikit-learn pillow

echo -e "\n[2/6] Ingesting local data..."
python data/get_data.py

echo -e "\n[3/6] Cleaning dataset (removing duplicates and corrupted files)..."
python data/cleaning.py

echo -e "\n[4/6] Splitting dataset into Train/Val/Test..."
python data/split.py

echo -e "\n[5/6] Training Top Layers and Fine-Tuning MobileNetV2..."
python models/finetune.py

echo -e "\n[6/6] Launching Traffic Sign Recognition GUI..."
python models/entrypoint.py

echo "\nPipeline execution finished!"
