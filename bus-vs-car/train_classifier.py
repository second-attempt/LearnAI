import os
import sys
from fastai.vision.all import *
from pathlib import Path
from PIL import Image, UnidentifiedImageError

# Use fallback if MPS (Apple Silicon) not supported
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

path = Path('data/Vehicles')

def remove_corrupt_images(folder):
    """
    Removes corrupted or unreadable images from the dataset.
    """
    print("🔍 Checking for corrupt images...")
    count = 0
    for img_path in Path(folder).rglob("*.jpg"):
        try:
            img = Image.open(img_path)
            img.verify()
        except (UnidentifiedImageError, OSError):
            print(f"Removing corrupted image: {img_path}")
            img_path.unlink()
            count += 1
    print(f"Corruption check complete. Removed {count} bad images.\n")

remove_corrupt_images(path)

files = get_image_files(path)
print(f"Total image files found: {len(files)}")

# Show some sample files and labels
print("\n Sample image files and their parent labels:")
for f in files[:10]:
    print(f" - {f} => {parent_label(f)}")
print()

# Create DataBlock
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(192, method='squish'),
    batch_tfms=[IntToFloatTensor()]
)

# Create Dataloaders
dls = dblock.dataloaders(path, bs=32)

# Validate the DataLoaders
try:
    batch = next(iter(dls.train))
    print(f"Dataloader sample batch loaded successfully.")
    print(f"Batch size: {len(batch)}; Image shape: {batch[0].shape}; Labels: {batch[1][:5]}")
except Exception as e:
    print(f"Failed to load a batch from DataLoader: {e}")
    sys.exit(1)

# Dataset sizes
print(f"\n Training set size: {len(dls.train_ds)}")
print(f"Validation set size: {len(dls.valid_ds)}")

if len(dls.train_ds) == 0 or len(dls.valid_ds) == 0:
    print("No data available for training or validation. Exiting.")
    sys.exit(1)

# Training
learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.unfreeze()

print("\n Starting training...")
learn.fine_tune(20, base_lr=1e-3, freeze_epochs=2)
print(" Training complete.")

# Validation
val_loss, val_acc = learn.validate()
print(f"\n Final Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# Save model
model_path = path/'bus_vs_car_model.pkl'
learn.export(model_path)
print(f"Model exported to: {model_path}")

# Inference
try:
    img = PILImage.create('InputImage.jpg')
    is_car, _, probs = learn.predict(img)
    print(f"\n🔍 Prediction result:")
    print(f"Label: {is_car}")
    print(f"Probability: {probs.max():.4f}")
except Exception as e:
    print(f"Could not predict due to: {e}")
