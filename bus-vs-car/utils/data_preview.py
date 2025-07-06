import random
from fastai.vision.all import *
from pathlib import Path

class DataPreviewer:
    @staticmethod
    def preview_sample_images(files, n=5):
        print("\n📸 Sample image files and their parent labels:")
        for f in random.sample(files, min(n, len(files))):
            print(f" - {f} => {parent_label(f)}")
        print()

    @staticmethod
    def show_dataset_summary(dls):
        print(f"\nTraining set size: {len(dls.train_ds)}")
        print(f"Validation set size: {len(dls.valid_ds)}")
        if len(dls.train_ds) == 0 or len(dls.valid_ds) == 0:
            raise ValueError("No data available for training or validation.")
