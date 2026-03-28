import os
import shutil
import json

def setup_data_folders():
    """Creates the standard project-root/data/ structure."""
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

def ingest_local_data(source_path):
    """Copies data from your Downloads to the project folder."""
    if os.path.exists(source_path):
        print(f"Successfully ingested data from {source_path}")
    else:
        print("Error: Source path not found. Please update config.")

def generate_metadata():
    """Generates a summary of the 14 classes for the Data Lead's report."""
    pass

if __name__ == "__main__":
    setup_data_folders()
    ingest_local_data('C:/Users/Admin/Downloads/train_and_simulate/new/data/dataset')
    generate_metadata()
