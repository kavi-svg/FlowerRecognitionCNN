# Flower Recognition CNN

A professional-grade convolutional neural network (CNN) for automated flower classification using TensorFlow and Keras. This project provides an end-to-end pipeline for training custom flower recognition models and deploying them for real-world predictions with detailed HTML reports.

## Features

- ✅ **Deep CNN Architecture** — Multi-layer convolutional model optimized for flower classification
- ✅ **Flexible Input** — Accepts local image files or remote HTTP(S) URLs
- ✅ **Professional Reports** — Beautiful HTML reports with confidence metrics and statistics
- ✅ **Easy Deployment** — Train once, predict on unlimited images
- ✅ **Error Handling** — Robust error handling and detailed logging
- ✅ **Production Ready** — Early stopping, model checkpointing, and data augmentation

## Quick Start (5 minutes)

### Prerequisites

- Python 3.11+ (TensorFlow compatibility)
- pip or conda

### 1. Setup environment

```powershell
# Create virtual environment
python -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate test images

```powershell
python create_test_images.py

# Create class folders (required)
New-Item -ItemType Directory -Path data\flowers\class1 -Force
New-Item -ItemType Directory -Path data\flowers\class2 -Force
```

### 3. Run predictions on test images

```powershell
python flower_cnn.py --model_path models/flower_cnn.h5 --predict_urls "C:\Users\hp\Downloads\flower1.jpg,C:\Users\hp\Downloads\flower2.jpg,C:\Users\hp\Downloads\flower3.jpg" --output_dir predictions
```

### 4. View professional report

```powershell
Start-Process .\predictions\report.html
```

You'll see:
- 📊 Prediction statistics (images processed, confidence metrics)
- 🖼️ Annotated images with classification labels
- 📈 Confidence scores with visual progress bars
- 🔗 Source image links for reference

## Training a production model

### Dataset preparation

Organize your flower images by class in `data/flowers/`:

```
data/flowers/
├── rose/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── daisy/
│   ├── img1.jpg
│   └── ...
└── sunflower/
    └── ...
```

**Recommended**: At least 50 images per class for good results.

### Train the model

```powershell
python flower_cnn.py --data_dir data/flowers --epochs 20 --model_path models/flower_cnn.h5
```

**Options:**
- `--epochs` — Number of training epochs (default: 12)
- `--batch_size` — Batch size for training (default: 32)
- `--image_size` — Input image resolution (default: 180x180)

### Monitor training

The model automatically:
- Splits data into 80% training, 20% validation
- Saves the best model (lowest validation loss)
- Stops early if validation loss plateaus (patience: 3 epochs)

## Prediction API

Run inference on any images (local or remote):

```powershell
# Single local file
python flower_cnn.py --model_path models/flower_cnn.h5 --predict_urls "C:\path\to\image.jpg"

# Multiple files (comma-separated)
python flower_cnn.py --model_path models/flower_cnn.h5 --predict_urls "img1.jpg,img2.jpg,img3.jpg"

# Remote HTTPS URL
python flower_cnn.py --model_path models/flower_cnn.h5 --predict_urls "https://example.com/flower.jpg"

# Mixed local + remote
python flower_cnn.py --model_path models/flower_cnn.h5 --predict_urls "local_image.jpg,https://example.com/remote.jpg"

# Custom output directory
python flower_cnn.py --model_path models/flower_cnn.h5 --predict_urls "image.jpg" --output_dir results/batch_1
```

## Output

Each prediction run generates:

**predictions/** (or custom `--output_dir`)
- `prediction_1.jpg` — Image with classification label overlay
- `prediction_2.jpg` — (and more for each input image)
- `report.html` — Professional HTML report with statistics

The HTML report includes:
- Total images processed
- Highest confidence score
- Average confidence across predictions
- Clickable links to source images
- Responsive grid layout for mobile

## Model Architecture

```
Input (180x180x3)
  ↓
Rescaling (1/255)
  ↓
Conv2D (32 filters) → ReLU → MaxPooling
  ↓
Conv2D (64 filters) → ReLU → MaxPooling
  ↓
Conv2D (128 filters) → ReLU → MaxPooling
  ↓
Dropout (0.3)
  ↓
Flatten
  ↓
Dense (128) → ReLU → Dropout (0.5)
  ↓
Dense (num_classes) → Softmax
  ↓
Output (class probabilities)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'tensorflow'" | `pip install tensorflow>=2.14.0` |
| Report is blank | Check that `data/flowers/class1` and `class2` folders exist |
| Image URL fails | Ensure URL points to a direct image file (.jpg, .png, etc.) |
| Slow predictions | GPU not available on Windows; use WSL2 or ignore |
| Permission denied | Run PowerShell as Administrator or adjust execution policy |

## File Structure

```
FlowerRecognitionCNN/
├── flower_cnn.py              # Main training & prediction script
├── create_test_images.py      # Generate sample test flowers
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── .gitignore                 # Git ignore rules
├── data/
│   └── flowers/               # Dataset folder
│       ├── class1/
│       └── class2/
├── models/
│   └── flower_cnn.h5          # Trained model (HDF5 format)
└── predictions/               # Output folder
    ├── report.html
    ├── prediction_1.jpg
    ├── prediction_2.jpg
    └── ...
```

## Performance Tips

- **Data quality**: Use high-resolution, well-lit images for best results
- **Data variety**: Include different angles, lighting, backgrounds
- **Augmentation**: TensorFlow automatically applies rescaling; consider manual augmentation for large datasets
- **Class balance**: Aim for roughly equal samples per flower class
- **Training time**: Expect 5-15 minutes on CPU depending on dataset size

## Requirements

- Python 3.11+
- TensorFlow 2.14+
- NumPy 1.24+
- Pillow 10.0+

See `requirements.txt` for exact versions.

## License

Open source for educational and commercial use.

## Support

For issues, questions, or improvements, please open an issue or contact the project maintainer.

---

**Built with TensorFlow & Keras** | Flower Recognition CNN
