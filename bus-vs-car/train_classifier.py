# train_classifier.py
import os
import sys
from fastai.vision.all import *
from pathlib import Path
from utils.image_cleaning import ImageCleaner
from utils.data_preview import DataPreviewer
from utils.validation import DataValidator
from utils.metrics import MetricsReporter

# Enable fallback for MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Define dataset path
path = Path('data/Vehicles')

# Step 1: Remove corrupt images
cleaner = ImageCleaner()
cleaner.remove_corrupt_images(path)

# Step 2: Preview sample images
files = get_image_files(path)
print(f"Total image files found: {len(files)}")
previewer = DataPreviewer()
previewer.preview_sample_images(files)

# Step 3: Prepare DataBlock and DataLoaders
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(192, method='squish'),
    batch_tfms=[IntToFloatTensor()]
)

dls = dblock.dataloaders(path, bs=32)
previewer.show_dataset_summary(dls)

# Step 4: Validate DataLoaders
validator = DataValidator()
validator.validate_dataloaders(dls)

# Step 5: Training
if len(dls.train_ds) == 0 or len(dls.valid_ds) == 0:
    print("No data available for training or validation. Exiting.")
    sys.exit(1)

learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.unfreeze()
print("\n Starting training...")
learn.fine_tune(20, base_lr=1e-3, freeze_epochs=2)
print(" Training complete.")

# Step 6: Analyze confusion matrix
interp = ClassificationInterpretation.from_learner(learn)
reporter = MetricsReporter()
reporter.show_confusion_matrix(interp, learn.dls.vocab)

# Step 7: Final validation
val_loss, val_acc = learn.validate()
print(f"\n Final Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# Step 8: Export model
model_path = path/'bus_vs_car_model.pkl'
learn.export(model_path)
print(f"Model exported to: {model_path}")

# Step 9: Inference
try:
    img = PILImage.create('InputImage.jpg')
    is_car, _, probs = learn.predict(img)
    print(f"\n🔍 Prediction result:")
    print(f"Label: {is_car}")
    print(f"Probability: {probs.max():.4f}")
except Exception as e:
    print(f"Could not predict due to: {e}")
