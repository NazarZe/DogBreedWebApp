import os
import time
import logging
import numpy as np
import pandas as pd
import pickle
import glob
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import ResNet50, MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Set up logging
logging.basicConfig(filename="pipeline_log.txt", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PARAMETERS
IMG_SIZE = (224, 224)
NUM_CLASSES = 10
BATCH_LIMIT_PER_CLASS = 100
DATA_DIR = "images"

# Load images and labels
def load_images_and_labels(base_dir, img_size, max_per_class):
    X, y, class_names = [], [], []
    class_dirs = sorted(os.listdir(base_dir))[:NUM_CLASSES]
    for label, class_name in enumerate(class_dirs):
        class_path = os.path.join(base_dir, class_name)
        images = glob.glob(os.path.join(class_path, "*.jpg"))[:max_per_class]
        class_names.append(class_name)
        for img_path in images:
            img = load_img(img_path, target_size=img_size)
            img = img_to_array(img)
            X.append(img)
            y.append(label)
    logging.info(f"Завантажено {len(X)} зображень з {len(class_names)} класів")
    return np.array(X), np.array(y), class_names

# Extract features
def extract_features(model, preprocess_fn, X):
    X_prep = preprocess_fn(X.copy())
    return model.predict(X_prep, verbose=0)

# Train classifiers
def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_name):
    results = []
    models = {
        "SVM": SVC(kernel='linear'),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    with open(f"scaler_{feature_name}.pkl", "wb") as f:
        pickle.dump(scaler, f)

    for name, model in models.items():
        model_path = f"model_{feature_name}_{name}.pkl"
        if not os.path.exists(model_path):
            model.fit(X_train, y_train)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        start = time.time()
        y_pred = model.predict(X_test)
        duration = time.time() - start
        acc = accuracy_score(y_test, y_pred)
        logging.info(f"{feature_name}-{name}: Accuracy={acc:.4f}, Time={duration:.2f}s")
        results.append((name, acc, duration))
    return results

# Predict all models manually
def predict_all_models(image_path, class_names, model_filter=None):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_arr = img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)

    results = []
    for cnn_name, (cnn_model, preprocess_fn) in cnn_models.items():
        if model_filter and cnn_name != model_filter:
            continue
        try:
            features = cnn_model.predict(preprocess_fn(img_arr), verbose=0)
            scaler_path = f"scaler_{cnn_name}.pkl"
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            features_scaled = scaler.transform(features)

            for model_name in ["SVM", "MLP", "RandomForest"]:
                model_path = f"model_{cnn_name}_{model_name}.pkl"
                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        model = pickle.load(f)
                    prediction = model.predict(features_scaled)
                    predicted_class = class_names[int(prediction[0])]
                    results.append((cnn_name, model_name, predicted_class))
        except Exception as e:
            results.append((cnn_name, "ERROR", str(e)))

    return results

# MAT model prediction (optional)
def predict_mat_model(image_path):
    try:
        with open("model_MAT_ExtraTrees.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler_mat.pkl", "rb") as f:
            scaler = pickle.load(f)

        flat_img = load_img(image_path, target_size=(100, 120))  # resize to match expected size
        flat_img = img_to_array(flat_img).flatten().reshape(1, -1)
        flat_img_scaled = scaler.transform(flat_img)
        prediction = model.predict(flat_img_scaled)
        return int(prediction[0])
    except Exception as e:
        return f"❗ Помилка MAT-класифікації: {e}"

# Plotting (optional, manual call)
def generate_plots(results_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="FeatureExtractor", y="Accuracy", hue="Model")
    plt.title("Model Accuracy by Feature Extractor")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="FeatureExtractor", y="TrainTime", hue="Model")
    plt.title("Training Time by Feature Extractor")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig("train_time_comparison.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="FeatureExtractor", y="FeatureGenTime")
    plt.title("Feature Extraction Time by Network")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig("feature_time_comparison.png")
    plt.close()

# Init (only executed on import, not on Flask call)
X_train_img = X_test_img = y_train = y_test = class_names = None
if os.path.exists(DATA_DIR):
    X, y, class_names = load_images_and_labels(DATA_DIR, IMG_SIZE, BATCH_LIMIT_PER_CLASS)
    X_train_img, X_test_img, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
else:
    class_names = [f"Class {i}" for i in range(NUM_CLASSES)]

cnn_models = {
    "ResNet50": (ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=IMG_SIZE + (3,)), resnet_preprocess),
    "MobileNetV2": (MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=IMG_SIZE + (3,)), mobilenet_preprocess),
    "EfficientNetB0": (EfficientNetB0(weights="imagenet", include_top=False, pooling="avg", input_shape=IMG_SIZE + (3,)), efficientnet_preprocess),
}


