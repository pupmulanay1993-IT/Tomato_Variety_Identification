import json
from PIL import Image
import numpy as np
import cv2
import skfuzzy as fuzz
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------------------------
# COLOR INTELLIGENCE DATABASE
# -------------------------------------------------
VARIETY_COLORS = {
    "apollo_tomato": {"ripe": "red", "overripe": "dark_red"},
    "atlas_tomato": {"ripe": "red", "overripe": "soft_dark_red"},
    "cherry_tomato": {"ripe": "bright_red_yellow", "overripe": "deep_orange"},
    "diamante_tomato": {"ripe": "orange_red", "overripe": "red"},
    "kinalabasa_tomato": {"ripe": "orange_red", "overripe": "deep_red"},
    "pear_tomato": {"ripe": "yellow", "overripe": "orange"},
    "rio_grande_tomato": {"ripe": "red", "overripe": "deep_red"},
    "roma_tomato": {"ripe": "red", "overripe": "dark_red"},
    "non_tomato": {"ripe": "none", "overripe": "none"}
}

# -------------------------------------------------
# ADAPTIVE COLOR SCORING
# -------------------------------------------------
def compute_color_scores(image, variety_label="Unknown"):
    if isinstance(image, Image.Image):
        img_np = np.array(image.convert("RGB"))
    else:
        img_np = np.array(image)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

    v_info = VARIETY_COLORS.get(variety_label, {"ripe": "red"})

    if v_info["ripe"] == "yellow":
        lower = np.array([20, 80, 80])
        upper = np.array([40, 255, 255])
    elif "orange" in v_info["ripe"]:
        lower = np.array([0, 50, 50])
        upper = np.array([25, 255, 255])
    else:
        lower = np.array([0, 70, 70])
        upper = np.array([15, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    hsv_percent = (np.count_nonzero(mask) / mask.size) * 100

    a_mean = np.mean(lab[:, :, 1])
    b_mean = np.mean(lab[:, :, 2])

    if v_info["ripe"] == "yellow":
        lab_score = fuzz.membership.gaussmf(b_mean, 190, 30)
    else:
        lab_score = fuzz.membership.gaussmf(a_mean, 160, 40)

    return float(hsv_percent), float(lab_score)

# -------------------------------------------------
# TOMATO PRESENCE CHECK
# -------------------------------------------------
def is_tomato_bouncer(image, min_tomato_percent=5):
    hsv_percent, _ = compute_color_scores(image)
    return hsv_percent >= min_tomato_percent

# -------------------------------------------------
# IMAGE PREPROCESSING
# -------------------------------------------------
def clean_image(image, target_size=(224, 224)):
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = np.array(image)

    denoised = cv2.fastNlMeansDenoisingColored(image_np, None, 5, 5, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_LANCZOS4)
    img_expanded = np.expand_dims(resized, axis=0).astype(np.float32)
    return preprocess_input(img_expanded)

# -------------------------------------------------
# COLOR CLASSIFICATION
# -------------------------------------------------
def classify_hsv_color(hsv_color, color_map):
    h, s, v = hsv_color
    h_deg = (h / 180.0) * 360.0

    if s < 40 or v < 40:
        return "Uncertain", 0.0, color_map.get("Other", "#999999")
    if 20 <= h_deg <= 40:
        return "Orange Tomato", s / 255.0, color_map.get("Orange Tomato", "#FF7F00")
    if 41 <= h_deg <= 60:
        return "Yellow Tomato", s / 255.0, color_map.get("Yellow Tomato", "#FFFF00")
    if h_deg < 20 or h_deg > 330:
        return "Red Tomato", s / 255.0, color_map.get("Red Tomato", "#FF0000")
    if 60 < h_deg <= 150:
        return "Green Tomato", s / 255.0, color_map.get("Green Tomato", "#00FF00")
    return "Other", s / 255.0, color_map.get("Other", "#999999")

# -------------------------------------------------
# MULTI COLOR DETECTION (FIXED)
# -------------------------------------------------
def detect_multi_colors(image_rgb, k=4, min_conf=0.3, color_map=None):
    if color_map is None:
        color_map = {}

    img_small = cv2.resize(image_rgb, (100, 100), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_small, cv2.COLOR_RGB2HSV)
    hsv_pixels = hsv.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(hsv_pixels)
    centers, counts = kmeans.cluster_centers_, np.bincount(kmeans.labels_)
    total = counts.sum()

    detected = {}
    for i, center in enumerate(centers):
        label, conf, hex_color = classify_hsv_color(center, color_map)
        weight = float(counts[i]) / float(total)
        if label != "Uncertain" and (conf * weight) > 0.05:
            detected[label] = {"confidence": conf * weight, "color": hex_color}

    return detected

# -------------------------------------------------
# MODEL PREDICTION
# -------------------------------------------------
def get_prediction(model, image_preprocessed):
    preds = model.predict(image_preprocessed, verbose=0)
    probs = preds[0]
    idx = int(np.argmax(probs))
    conf = float(np.max(probs))
    return probs, idx, conf

# -------------------------------------------------
# RESULT FORMATTER
# -------------------------------------------------
def make_results(avg_preds, indices, confs, class_indices_path="class_indices.json"):
    try:
        with open(class_indices_path, "r") as f:
            class_mapping = json.load(f)
        class_labels = {int(v): k for k, v in class_mapping.items()}
    except:
        class_labels = {
            0: "apollo_tomato", 1: "atlas_tomato", 2: "cherry_tomato",
            3: "diamante_tomato", 4: "kinalabasa_tomato", 5: "non_tomato",
            6: "pear_tomato", 7: "rio_grande_tomato", 8: "roma_tomato"
        }

    recs = {
        "apollo_tomato": {
            "description": "Hybrid variety known for high yield and heat tolerance. Oval-round fruit.",
            "plant_lifespan": "Annual (Determinate)",
            "shelf_life": {"room_temp_days": 3, "refrigerated_days": 9},
            "temperature_feasibility": {"ideal_temp_c": [20, 30], "feasibility_note": "Best in sunny, low-land areas."}
        },
        "atlas_tomato": {
            "description": "Large, meaty fruit variety, often used for fresh salads.",
            "plant_lifespan": "Annual (Indeterminate)",
            "shelf_life": {"room_temp_days": 4, "refrigerated_days": 10},
            "temperature_feasibility": {"ideal_temp_c": [18, 30], "feasibility_note": "Requires staking for support."}
        },
        "cherry_tomato": {
            "description": "Small, bite-sized tomatoes with high sugar content.",
            "plant_lifespan": "Annual (Indeterminate)",
            "shelf_life": {"room_temp_days": 5, "refrigerated_days": 12},
            "temperature_feasibility": {"ideal_temp_c": [18, 28], "feasibility_note": "Fast growing and productive."}
        },
        "diamante_tomato": {
            "description": "Highly resistant to pests and diseases, firm fruit quality.",
            "plant_lifespan": "Annual",
            "shelf_life": {"room_temp_days": 3, "refrigerated_days": 9},
            "temperature_feasibility": {"ideal_temp_c": [22, 32], "feasibility_note": "Excellent heat tolerance."}
        },
        "kinalabasa_tomato": {
            "description": "Traditional Filipino variety with a flat, ribbed shape (pumpkin-like).",
            "plant_lifespan": "Annual",
            "shelf_life": {"room_temp_days": 3, "refrigerated_days": 10},
            "temperature_feasibility": {"ideal_temp_c": [20, 32], "feasibility_note": "Very hardy in local weather."}
        },
        "pear_tomato": {
            "description": "Unique pear-shaped fruit, usually yellow or red. Sweet flavor.",
            "plant_lifespan": "Annual",
            "shelf_life": {"room_temp_days": 5, "refrigerated_days": 12},
            "temperature_feasibility": {"ideal_temp_c": [20, 30], "feasibility_note": "Great for snacks and garnishes."}
        },
        "rio_grande_tomato": {
            "description": "Blocky, pear-shaped fruit. Thick walls make it perfect for sauces.",
            "plant_lifespan": "Annual (Determinate)",
            "shelf_life": {"room_temp_days": 4, "refrigerated_days": 10},
            "temperature_feasibility": {"ideal_temp_c": [18, 28], "feasibility_note": "Resistant to verticillium wilt."}
        },
        "roma_tomato": {
            "description": "Egg-shaped tomato with few seeds. The standard for tomato paste.",
            "plant_lifespan": "Annual (Determinate)",
            "shelf_life": {"room_temp_days": 4, "refrigerated_days": 11},
            "temperature_feasibility": {"ideal_temp_c": [18, 30], "feasibility_note": "Very consistent growth."}
        },
        "non_tomato": {
            "description": "The image provided does not look like a valid tomato variety.",
            "plant_lifespan": "N/A",
            "shelf_life": {"room_temp_days": 0, "refrigerated_days": 0},
            "temperature_feasibility": {"ideal_temp_c": [0,0], "feasibility_note": "N/A"}
        }
    }

    class_name = class_labels.get(indices, "Unknown")
    recommendation_data = recs.get(class_name, {"description": "Details coming soon."})

    return {
        "status": f"Detected: {class_name.replace('_', ' ').title()}",
        "variety_label": class_name,
        "prediction": f"{int(confs * 100)}%",
        "recommendation": recommendation_data
    }
