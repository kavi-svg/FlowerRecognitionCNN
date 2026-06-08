import argparse
import io
import pathlib
import re
import sys
import traceback
import urllib.request
from datetime import datetime

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow import keras
from tensorflow.keras import layers


def prepare_datasets(data_dir, image_size=(180, 180), batch_size=32, validation_split=0.2, seed=123):
    data_dir = pathlib.Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def get_class_names(data_dir):
    data_dir = pathlib.Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    return [path.name for path in sorted(data_dir.iterdir()) if path.is_dir()]


def sanitize_filename(name):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name[:200]


def download_image(url):
    """Load image from local file path or remote URL (HTTP/HTTPS/file://)."""
    p = pathlib.Path(url)
    if p.is_file():
        try:
            return Image.open(str(p)).convert("RGB")
        except Exception as e:
            print(f"    [local file] Error opening {p}: {e}")
            raise

    if url.startswith("file://"):
        local = url[7:]
        p = pathlib.Path(local)
        if p.is_file():
            try:
                return Image.open(str(p)).convert("RGB")
            except Exception as e:
                print(f"    [file:// URL] Error opening {p}: {e}")
                raise

    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            image_data = response.read()
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        print(f"    [HTTP URL] Error fetching {url}: {e}")
        raise


def preprocess_image(image, image_size=(180, 180)):
    image = image.resize(image_size)
    array = np.array(image, dtype=np.float32) / 255.0
    return array


def save_prediction_image(image, label, score, output_dir, index):
    """Save image with prediction label drawn on top."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"{label} ({score:.1%})"

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = 8
    rect = [0, 0, text_width + padding * 2, text_height + padding * 2]
    draw.rectangle(rect, fill=(0, 0, 0, 200))
    draw.text((padding, padding), text, fill="white", font=font)

    safe_name = sanitize_filename(f"prediction_{index}.jpg")
    output_path = output_dir / safe_name
    image.save(output_path, "JPEG", quality=90)
    return output_path


def save_html_report(entries, report_path):
    """Save prediction results to a professional HTML report."""
    report_path = pathlib.Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    html = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"UTF-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">",
        "<title>Flower Classification Report</title>",
        "<style>",
        "* { margin: 0; padding: 0; box-sizing: border-box; }",
        "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; color: #333; }",
        ".container { max-width: 1200px; margin: 0 auto; padding: 40px 20px; }",
        ".header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 10px; margin-bottom: 40px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }",
        ".header h1 { font-size: 2.5em; margin-bottom: 10px; }",
        ".header p { font-size: 1.1em; opacity: 0.95; }",
        ".stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }",
        ".stat-card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }",
        ".stat-card .number { font-size: 2.5em; font-weight: bold; color: #667eea; }",
        ".stat-card .label { color: #666; margin-top: 8px; font-size: 0.95em; }",
        ".predictions { display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 25px; }",
        ".prediction-card { background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.3s, box-shadow 0.3s; }",
        ".prediction-card:hover { transform: translateY(-5px); box-shadow: 0 8px 20px rgba(0,0,0,0.15); }",
        ".prediction-image { width: 100%; height: 250px; object-fit: cover; }",
        ".prediction-info { padding: 20px; }",
        ".prediction-label { font-size: 1.4em; font-weight: 600; color: #333; margin-bottom: 10px; }",
        ".prediction-confidence { display: flex; align-items: center; gap: 10px; margin-bottom: 15px; }",
        ".confidence-bar { flex: 1; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; }",
        ".confidence-fill { height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }",
        ".confidence-text { font-weight: 600; color: #667eea; min-width: 50px; text-align: right; }",
        ".prediction-url { font-size: 0.85em; color: #0066cc; word-break: break-all; text-decoration: none; }",
        ".prediction-url:hover { text-decoration: underline; }",
        ".footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 0.9em; }",
        "</style>",
        "</head>",
        "<body>",
        "<div class=\"container\">",
        "<div class=\"header\">",
        "<h1>🌸 Flower Classification Report</h1>",
        "<p>AI-powered flower recognition using CNN</p>",
        "</div>",
    ]

    if entries:
        html.extend([
            "<div class=\"stats\">",
            f"<div class=\"stat-card\"><div class=\"number\">{len(entries)}</div><div class=\"label\">Images Processed</div></div>",
            f"<div class=\"stat-card\"><div class=\"number\">{max(e[3] for e in entries):.1%}</div><div class=\"label\">Highest Confidence</div></div>",
            f"<div class=\"stat-card\"><div class=\"number\">{sum(e[3] for e in entries) / len(entries):.1%}</div><div class=\"label\">Average Confidence</div></div>",
            "</div>",
            "<div class=\"predictions\">",
        ])
        for idx, (image_path, url, label, score) in enumerate(entries, 1):
            html.append(
                f"<div class=\"prediction-card\">"
                f"<img src=\"{image_path.name}\" alt=\"Prediction {idx}\" class=\"prediction-image\">"
                f"<div class=\"prediction-info\">"
                f"<div class=\"prediction-label\">{label}</div>"
                f"<div class=\"prediction-confidence\">"
                f"<div class=\"confidence-bar\"><div class=\"confidence-fill\" style=\"width: {score * 100:.1f}%\"></div></div>"
                f"<div class=\"confidence-text\">{score:.1%}</div>"
                f"</div>"
                f"<a href=\"{url}\" class=\"prediction-url\" target=\"_blank\">View source</a>"
                f"</div>"
                f"</div>"
            )
        html.append("</div>")
    else:
        html.extend([
            "<div style=\"background: white; padding: 40px; text-align: center; border-radius: 10px;\">",
            "<h2 style=\"color: #999; margin-bottom: 10px;\">No Predictions</h2>",
            "<p style=\"color: #aaa;\">No successful predictions were made. Check image URLs and ensure class folders exist.</p>",
            "</div>",
        ])

    html.extend([
        "</div>",
        "<div class=\"footer\">",
        f"<p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "<p>Flower Recognition CNN | TensorFlow-based Classification</p>",
        "</div>",
        "</body>",
        "</html>",
    ])
    report_path.write_text("\n".join(html), encoding="utf-8")
    return report_path


def predict_urls(model, class_names, urls, image_size=(180, 180), output_dir="predictions"):
    """Run predictions on images and save results to HTML report."""
    if isinstance(urls, str):
        urls = [u.strip() for u in urls.split(",") if u.strip()]

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Flower Prediction ===")
    print(f"URLs to process: {len(urls)}")
    print(f"Class names: {class_names}")
    print(f"Output directory: {output_dir}")

    entries = []
    for index, url in enumerate(urls, start=1):
        print(f"\n[{index}/{len(urls)}] {url[:80]}...")
        try:
            print(f"  [1] Loading image...")
            original_image = download_image(url)
            print(f"  [2] Image loaded: {original_image.size}")

            print(f"  [3] Preprocessing...")
            image = preprocess_image(original_image, image_size=image_size)
            print(f"  [4] Running model prediction...")
            image_batch = np.expand_dims(image, axis=0)
            prediction = model.predict(image_batch, verbose=0)[0]
            predicted_idx = int(np.argmax(prediction))
            label = class_names[predicted_idx]
            score = float(np.max(prediction))
            print(f"  [5] Prediction: {label} ({score:.1%})")

            print(f"  [6] Saving annotated image...")
            saved_path = save_prediction_image(original_image, label, score, output_dir, index)
            entries.append((saved_path, url, label, score))
            print(f"  ✓ SUCCESS")
        except Exception as exc:
            print(f"  ✗ FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    report_path = output_dir / "report.html"
    save_html_report(entries, report_path)
    print(f"\n=== Results ===")
    print(f"Total successful: {len(entries)}/{len(urls)}")
    print(f"Report: {report_path}")
    if entries:
        print(f"Images saved to: {output_dir}")
        for _, _, label, score in entries:
            print(f"  - {label} ({score:.1%})")
    else:
        print("WARNING: No predictions were successful. Check URLs and class folders.")


def build_model(num_classes, input_shape=(180, 180, 3)):
    model = keras.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Train or predict with a flower classification CNN.")
    parser.add_argument("--data_dir", default="data/flowers", help="Path to image folder with subfolders per class.")
    parser.add_argument("--image_size", type=int, default=180, help="Height and width to resize images.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs.")
    parser.add_argument("--model_path", default="models/flower_cnn.h5", help="Where to save or load the model.")
    parser.add_argument("--predict_urls", default="", help="Comma-separated URLs of images to classify.")
    parser.add_argument("--output_dir", default="predictions", help="Directory where prediction images and the HTML report are saved.")
    parser.add_argument("--flower_names", default="", help="Comma-separated custom flower names (e.g., 'Rose,Daisy,Sunflower').")
    args = parser.parse_args()

    model_path = pathlib.Path(args.model_path)
    if args.predict_urls:
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found: {model_path}")

        class_names = get_class_names(args.data_dir)

        if args.flower_names:
            custom_names = [name.strip() for name in args.flower_names.split(",")]
            if len(custom_names) != len(class_names):
                print(f"WARNING: Found {len(class_names)} classes but provided {len(custom_names)} flower names.")
                print(f"Classes: {class_names}")
                print(f"Provided names: {custom_names}")
                class_names = custom_names[:len(class_names)] + class_names[len(custom_names):]
            else:
                class_names = custom_names

        model = keras.models.load_model(model_path)
        predict_urls(
            model,
            class_names,
            args.predict_urls,
            image_size=(args.image_size, args.image_size),
            output_dir=args.output_dir,
        )
        return

    train_ds, val_ds, class_names = prepare_datasets(
        args.data_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
    )

    model = build_model(num_classes=len(class_names), input_shape=(args.image_size, args.image_size, 3))
    model.summary()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(model_path), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print("Training completed. Classes:", class_names)
    print(f"Best model saved to: {args.model_path}")

    return history


if __name__ == "__main__":
    main()
