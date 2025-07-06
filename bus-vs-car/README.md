# 🚗 Bus vs Car Image Classifier (FastAI)

This project uses the **FastAI** library to build and train an image classification model that can distinguish between **buses** and **cars**. It includes data cleaning, preprocessing, model training, evaluation, and inference steps — all modularized into reusable utility components. 

## 📁 Course

https://course.fast.ai/Lessons/lesson1.html

---

## 📁 Project Structure

```
.
├── data/
│   └── Vehicles/             # Folder containing car and bus images in subfolders
├── utils/
│   ├── image_cleaning.py     # Removes corrupted images
│   ├── data_preview.py       # Displays sample image previews
│   ├── validation.py         # Validates the data loaders
│   └── metrics.py            # Displays confusion matrix and metrics
├── InputImage.jpg            # Sample image used for inference
├── train_classifier.py       # Main training pipeline
└── README.md
```

---

## 📦 Requirements

* Python 3.8+
* macOS with Apple Silicon (or fallback CPU mode)
* Recommended: virtual environment

Install dependencies:

```bash
pip install fastai fastdownload duckduckgo-search pillow
```

---

## 📸 Dataset Format

Place your image dataset under `data/Vehicles/`:

```
data/Vehicles/
├── bus/
│   ├── car_on_road_1.jpg
│   ├── ...
├── car/
│   ├── bus_on_road_1.jpg
│   ├── ...
```

---

## 🚀 Running the Classifier

1. Clean corrupted images
2. Preview a few image samples
3. Train a ResNet34-based model
4. View confusion matrix and evaluation metrics
5. Export the model
6. Perform inference on `InputImage.jpg`

To run everything:

```bash
python train_classifier.py
```

---

## 🧠 Model Details

* **Model**: `ResNet34` from FastAI
* **Image Size**: 192x192 (resized with `squish`)
* **Batch Size**: 32
* **Metrics**: Accuracy
* **Training**: 20 epochs with learning rate `1e-3`

---

## 📈 Output

* Confusion matrix
* Accuracy and misclassification stats
* Final validation loss and accuracy
* Exported model at: `data/Vehicles/bus_vs_car_model.pkl`

---

## 🧪 Inference

Drop a test image as `InputImage.jpg` and get predictions printed to the console:

```bash
Label: car
Probability: 0.97
```

---

## 📬 Contact

Feel free to contribute or reach out for improvements or extensions (e.g., adding more classes, using different architectures).