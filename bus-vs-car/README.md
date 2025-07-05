# LearnAI
This folder contains code which I am using to learn AI

download_images.py
 Downloads, validates, and prepares images for ML training.
    
    - Downloads images via DuckDuckGo
    - Removes corrupt images
    - Converts to RGB
    - Resizes to img_size x img_size

    Args:
        search_term (str): The search query.
        folder (Path): Folder to save valid images.
        max_images (int): Max number of images to download.
        img_size (int): Final image size (square).

train_classifier.py
  This script trains a deep learning image classifier using FastAI to distinguish between buses and cars based on road scene images.