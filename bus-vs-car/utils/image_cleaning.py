from pathlib import Path
from PIL import Image, UnidentifiedImageError
import os

class ImageCleaner:
    @staticmethod
    def remove_corrupt_images(folder_path: Path):
        print("🔍 Checking for corrupt images...")
        count = 0
        for img_path in folder_path.rglob("*.jpg"):
            try:
                img = Image.open(img_path)
                img.verify()
            except (UnidentifiedImageError, OSError):
                print(f"Removing corrupted image: {img_path}")
                img_path.unlink()
                count += 1
        print(f"Corruption check complete. Removed {count} bad images.\n")
