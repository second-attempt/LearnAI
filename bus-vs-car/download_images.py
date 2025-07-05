import os
from duckduckgo_search import DDGS
from fastdownload import download_url
from pathlib import Path
from PIL import Image, UnidentifiedImageError

def download_and_prepare_images(search_term, folder, max_images=10, img_size=400):
    """
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
    """
    os.makedirs(folder, exist_ok=True)
    downloaded = 0
    try:
        with DDGS() as ddgs:
            results = ddgs.images(search_term, max_results=max_images*2)  # extra buffer
            if not results:
                print(f"No results for '{search_term}'")
                return

            for idx, result in enumerate(results):
                if downloaded >= max_images:
                    break

                url = result.get('image')
                if not url:
                    continue

                filename = f"{search_term.replace(' ', '_')}_{idx+1}.jpg"
                dest = folder / filename

                try:
                    # Download image
                    download_url(url, dest, show_progress=False)

                    # Verify and open again after verify
                    with Image.open(dest) as img:
                        img.verify()

                    with Image.open(dest) as img:
                        img = img.convert("RGB")
                        img = img.resize((img_size, img_size))  # Uniform square resize
                        img.save(dest, format="JPEG")
                    
                    print(f"Saved: {dest}")
                    downloaded += 1

                except (UnidentifiedImageError, OSError, ValueError) as e:
                    print(f"Corrupt or invalid image: {dest} | Reason: {e}")
                    if dest.exists():
                        dest.unlink()

                except Exception as e:
                    print(f"Failed to process image {idx+1}: {e}")
                    if dest.exists():
                        dest.unlink()

    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    base = Path("data/Vehicles")
    download_and_prepare_images("bus on road", base / "bus", max_images=30, img_size=400)
    download_and_prepare_images("car on road", base / "car", max_images=30, img_size=400)
