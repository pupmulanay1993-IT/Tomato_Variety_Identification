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
# SHELF LIFE ADJUSTMENT (Ripeness-aware)
# -------------------------------------------------
def adjust_shelf_life_for_ripeness(shelf_life, ripeness_score):
    """Adjust shelf-life days by ripeness:
    - Hilaw (green): longer longevity
    - Hinog (orange/yellow): baseline
    - Overripe (red): shorter longevity
    """
    if not shelf_life or not isinstance(shelf_life, dict):
        return shelf_life

    room_days = int(shelf_life.get("room_temp_days", 0))
    fridge_days = int(shelf_life.get("refrigerated_days", 0))

    if ripeness_score < 40:  # green/hilaw
        room_days = max(1, room_days + 9)
        fridge_days = max(1, fridge_days + 17)
    elif ripeness_score >= 75:  # overripe/lanta
        room_days = max(1, room_days - 1)
        fridge_days = max(1, fridge_days - 2)
    # else ripe/hinog -> baseline values

    return {"room_temp_days": room_days, "refrigerated_days": fridge_days}

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
            "description": "Early-fruiting, vigorous Australian hybrid. Produces large, fleshy, low-acid round red fruits up to 300g.",
            "plant_lifespan": "Indeterminate: 6–8 months. Fruits continuously over several months.",
            "shelf_life": {"room_temp_days": 3, "refrigerated_days": 9},
            "temperature_feasibility": {"ideal_temp_c": [20, 30], "feasibility_note": "Mild & Moderate Climates: Thrives in traditional Mediterranean or spring/summer conditions. Struggles significantly under harsh tropical heat waves."}
        },
        "atlas_tomato": {
            "description": "Tropical F1 hybrid widely grown in the Philippines. Highly resistant to Tomato Yellow Leaf Curl Virus (TYLCV). Produces large, oval, orange-red fruits.",
            "plant_lifespan": "Semi-Determinate: 5–7 months. Features an extended, prolonged harvesting life.",
            "shelf_life": {"room_temp_days": 4, "refrigerated_days": 10},
            "temperature_feasibility": {"ideal_temp_c": [18, 30], "feasibility_note": "Hot & Humid Lowlands: Specifically bred for tropical environments. It handles high night temperatures remarkably well without dropping its blossoms."}
        },
        "cherry_tomato": {
            "description": "Grouping of small, bite-sized round tomatoes. Highly prolific, intensely sweet, and juicy.",
            "plant_lifespan": "Indeterminate (mostly): 6–8 months. Produces until cold weather hits.",
            "shelf_life": {"room_temp_days": 5, "refrigerated_days": 12},
            "temperature_feasibility": {"ideal_temp_c": [18, 28], "feasibility_note": "Broadly Adaptable: Because the fruits are small, they require less energy to mature, allowing them to successfully set fruit even when temperatures fluctuate outside the ideal zone."}
        },
        "diamante_tomato": {
            "description": "Blockbuster East-West Seed F1 hybrid in Southeast Asia. Deep red, square-round, heavy fruits. Famous for wet-season adaptability.",
            "plant_lifespan": "Determinate: 4–5 months. Short, compact bush that yields a massive, concentrated harvest.",
            "shelf_life": {"room_temp_days": 3, "refrigerated_days": 9},
            "temperature_feasibility": {"ideal_temp_c": [22, 32], "feasibility_note": "Hot, Wet, & Tropical: A powerhouse for tropical farming. It is highly resistant to 'blossom drop' caused by oppressive, humid summer nights."}
        },
        "kinalabasa_tomato": {
            "description": "Rare Philippine heirloom named for its unique, deeply ribbed/ruffled shape resembling a pumpkin (kalabasa). Juicy, well-balanced flavor.",
            "plant_lifespan": "Indeterminate: 5–7 months. Traditional vine that continues fruiting under good care.",
            "shelf_life": {"room_temp_days": 3, "refrigerated_days": 10},
            "temperature_feasibility": {"ideal_temp_c": [20, 32], "feasibility_note": "Native Tropical Humid: As a traditional Philippine heirloom, it is naturally adapted to consistent warmth and high humidity, though it prefers partial afternoon shade during peak summer heat."}
        },
        "pear_tomato": {
            "description": "Small, teardrop-shaped heirloom variety (usually yellow or red). Mildly sweet, low-seed count, crisp texture.",
            "plant_lifespan": "Indeterminate: 6–8 months. Vigorous vines requiring tall staking.",
            "shelf_life": {"room_temp_days": 5, "refrigerated_days": 12},
            "temperature_feasibility": {"ideal_temp_c": [20, 30], "feasibility_note": "Cooler to Warm Temperate: Prefers steady, moderate summer warmth. Extreme heat waves (above 33°C) cause the plant to focus on survival rather than fruiting."}
        },
        "rio_grande_tomato": {
            "description": "Heavily productive open-pollinated plum variety. Large, blocky pear-shaped fruits ideal for processing, pastes, and sauces.",
            "plant_lifespan": "Determinate: 4–5 months. Compact bush that concentrates its yield efficiently.",
            "shelf_life": {"room_temp_days": 4, "refrigerated_days": 10},
            "temperature_feasibility": {"ideal_temp_c": [18, 28], "feasibility_note": "Highly Adaptable / Dry Heat: Outstanding performance in regions with hot days and cool nights. Its robust nature lets it handle dry heat exceptionally well."}
        },
        "roma_tomato": {
            "description": "The classic Italian plum tomato. Slender, firm, oblong fruit with low moisture and few seeds—the gold standard for canning and cooking.",
            "plant_lifespan": "Determinate: 4–5 months. Compact 'bush' structure that sets its crop over a short window.",
            "shelf_life": {"room_temp_days": 4, "refrigerated_days": 11},
            "temperature_feasibility": {"ideal_temp_c": [18, 30], "feasibility_note": "Warm & Dry (Arid): Thrives in warm, open sunshine with low humidity. High temperatures combined with stagnant, humid air make it highly susceptible to fungal issues like early blight."}
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
