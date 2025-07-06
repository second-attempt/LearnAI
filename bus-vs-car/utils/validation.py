import random
import sys

class DataValidator:
    @staticmethod
    def validate_dataloaders(dls):
        print("\n🔍 Validating DataLoaders...")
        try:
            indices = random.sample(range(len(dls.train_ds)), 5)
            print("Class index → Class name mapping:")
            for i, v in enumerate(dls.vocab):
                print(f"{i} → {v}")

            print("\nSample image file → category:")
            for idx in indices:
                img_path = dls.train_ds.items[idx]
                img, label_idx = dls.train_ds[idx]
                label_str = dls.vocab[label_idx]
                print(f"{img_path.name} → {label_idx} → {label_str}")
        except Exception as e:
            print(f"Failed to load a batch from DataLoader: {e}")
            sys.exit(1)
