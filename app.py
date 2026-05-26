import streamlit as st
import os

# -------------------------------------------------
# 1. PAGE CONFIG & LIBRARY IMPORTS
# -------------------------------------------------
st.set_page_config(page_title="Tomato Variety Identification", layout="wide", page_icon="favicon.png")

# QUICK INITIALIZATION 
if "show_predictions" not in st.session_state:
    st.session_state.show_predictions = False
if "show_help" not in st.session_state:
    st.session_state.show_help = False

from PIL import Image
import io
import numpy as np
import uuid
import tensorflow as tf
import cv2
import base64
import json
import pandas as pd
from supabase import create_client, Client
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Local utilities and functions
try:
    from utils import (
        clean_image,
        get_prediction,
        make_results,
        is_tomato_bouncer,
        detect_multi_colors,
        compute_color_scores,
        adjust_shelf_life_for_ripeness,
    )
except Exception as e:
    st.error(f"Error importing utils.py: {e}")

# -------------------------------------------------
# 2. ADAPTIVE FUZZY LOGIC ENGINE DEFINITION
# -------------------------------------------------
intensity = ctrl.Antecedent(np.arange(0, 101, 1), 'intensity')
accuracy = ctrl.Antecedent(np.arange(0, 101, 1), 'accuracy')
ripeness = ctrl.Consequent(np.arange(0, 101, 1), 'ripeness')

intensity.automf(3, names=['low', 'medium', 'high'])
accuracy.automf(3, names=['poor', 'average', 'good'])

ripeness['unripe'] = fuzz.trimf(ripeness.universe, [0, 0, 45])
ripeness['ripe'] = fuzz.trimf(ripeness.universe, [35, 65, 85])
ripeness['overripe'] = fuzz.trimf(ripeness.universe, [75, 100, 100])

rule1 = ctrl.Rule(intensity['low'], ripeness['unripe'])
rule2 = ctrl.Rule(accuracy['poor'], ripeness['unripe']) 
rule3 = ctrl.Rule(intensity['medium'] & accuracy['good'], ripeness['ripe'])
rule4 = ctrl.Rule(intensity['high'] & accuracy['good'], ripeness['ripe'])
rule5 = ctrl.Rule(intensity['high'] & accuracy['average'], ripeness['overripe'])

ripeness_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
ripeness_sim = ctrl.ControlSystemSimulation(ripeness_ctrl)

# -------------------------------------------------
# 3. PERFORMANCE CACHING & DB CONNECTION
# -------------------------------------------------
@st.cache_resource
def load_tomato_model():
    """
    Iniloload ang TFLite Model gamit ang tflite_runtime interpreter.
    Sobrang tipid nito sa RAM kumpara sa buong TensorFlow library.
    """
    try:
        if os.path.exists("tomato_model.tflite"):
            # Gumawa ng interpreter instance para sa .tflite file
            interpreter = tflite.Interpreter(model_path="tomato_model.tflite")
            interpreter.allocate_tensors()
            return interpreter
        else:
            st.error("tomato_model.tflite file not found in repository!")
            return None
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None

@st.cache_resource
def init_supabase() -> Client:
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception as e:
        return None

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        if os.path.exists(bin_file):
            with open(bin_file, "rb") as f:
                return base64.b64encode(f.read()).decode()
        return ""
    except:
        return ""

supabase = init_supabase()
model = load_tomato_model()
models = [model] if model else []

try:
    with open("class_indices.json", "r") as f:
        class_mapping = json.load(f)
    idx_to_label = {int(v): k for k, v in class_mapping.items()}
except:
    idx_to_label = {0: "apollo_tomato", 1: "atlas_tomato", 2: "cherry_tomato", 3: "diamante_tomato", 
                    4: "kinalabasa_tomato", 5: "non_tomato", 6: "pear_tomato", 7: "rio_grande_tomato", 8: "roma_tomato"}

def convert_to_serializable(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    return obj

def fetch_all_predictions():
    if not supabase: return None
    try:
        response = supabase.table("tomato_logs").select("*").execute()
        return response.data if response.data else []
    except:
        return None

def convert_predictions_to_excel(predictions):
    if not predictions: return None
    flattened_data = []
    for pred in predictions:
        row = {
            "ID": pred.get("id"),
            "Variety Label": pred.get("variety_label"),
            "Prediction": pred.get("prediction"),
            "Status": pred.get("status"),
            "HSV Score": pred.get("hsv_score"),
            "Lab Score": pred.get("lab_score"),
            "Fuzzy Ripeness": pred.get("fuzzy_ripeness"),
            "Source": pred.get("source"),
            "Created At": pred.get("created_at")
        }
        flattened_data.append(row)
    df = pd.DataFrame(flattened_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)
    output.seek(0)
    return output.getvalue()

# -------------------------------------------------
# 4. PREDICTION LOGIC (VARIETY-AWARE)
# -------------------------------------------------
def _get_model_input_size(model, fallback=(224, 224)):
    try:
        if hasattr(model, "inputs") and model.inputs:
            ishape = model.inputs[0].shape
            h, w = int(ishape[1]), int(ishape[2])
            if h and w: return (h, w)
    except: pass
    return fallback

def run_prediction(pil_image):
    img_rgb = np.array(pil_image.convert("RGB"))
    preds_list = []
    
    if not models:
        st.error("No model loaded to perform prediction.")
        return None, None

    for m in models:
        h, w = _get_model_input_size(m)
        img_clean = clean_image(pil_image, target_size=(h, w))
        preds, indices, confs = get_prediction(m, img_clean)
        preds_list.append(preds)

    avg_preds = np.mean(preds_list, axis=0)
    idx = int(np.argmax(avg_preds))
    conf = float(np.max(avg_preds))
    detected_variety = idx_to_label.get(idx, "Unknown")

    hsv_percent, lab_score = compute_color_scores(pil_image, variety_label=detected_variety)
    tomato_like = is_tomato_bouncer(pil_image)

    try:
        ripeness_sim.input['intensity'] = hsv_percent
        ripeness_sim.input['accuracy'] = lab_score * 100 if lab_score <= 1.0 else lab_score
        ripeness_sim.compute()
        fuzzy_score = ripeness_sim.output['ripeness']
    except:
        fuzzy_score = 0

    result = make_results(avg_preds, idx, conf, class_indices_path="class_indices.json")
    result.update({
        "variety_label": detected_variety,
        "prediction": float(conf),
        "prediction_display": f"{int(conf * 100)}%",
        "hsv_percent": float(hsv_percent),
        "lab_score": float(lab_score),
        "fuzzy_ripeness": float(fuzzy_score),
        "status": "Valid" if tomato_like else "Low Color Match"
    })

    color_map = {"Red Tomato": "#FF0000", "Orange Tomato": "#FF7F00", "Yellow Tomato": "#FFFF00", "Green Tomato": "#00FF00", "Other": "#999999"}
    res_colors = detect_multi_colors(img_rgb, k=4, color_map=color_map)

    return result, res_colors

# -------------------------------------------------
# 5. STYLING & HEADER (SAFE RENDERING)
# -------------------------------------------------
background_base64 = get_base64_of_bin_file("background.jpg")
logo_left_base64 = get_base64_of_bin_file("PUP Mulanay left.png")
logo_right_base64 = get_base64_of_bin_file("PUP Mulanay right.png")

st.markdown(
    f"""
<style>
/* 1. BASE APP & BACKGROUND */
.stApp {{
    background-image: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)), 
                      url("data:image/jpg;base64,{background_base64}");
    background-size: cover;
    background-position: center;
    color: #FFFFFF !important;
}}

/* 2. HEADER & LOGO STYLING - LARGE & PROMINENT */
.header-container {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    margin-bottom: 40px;
    background: linear-gradient(135deg, rgba(255, 215, 0, 0.15) 0%, rgba(34, 139, 34, 0.15) 100%);
    border-radius: 15px;
    border: 3px solid #FFD700;
    box-shadow: 0 8px 32px rgba(255, 215, 0, 0.3);
}}

.logo-img {{
    width: clamp(60px, 15vw, 100px);
    height: auto;
    filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.8));
}}

.header-text {{
    font-size: clamp(2.2rem, 7vw, 3.5rem); 
    font-weight: 900;
    color: #FFD700 !important;
    text-shadow: 3px 3px 6px #000000, 0px 0px 20px rgba(255, 215, 0, 0.5);
    text-align: center;
    flex-grow: 1;
    line-height: 1.3;
    letter-spacing: 2px;
}}

/* 3. SUBHEADERS - LARGER FONT */
.stSubheader {{
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #FFD700 !important;
    text-shadow: 2px 2px 4px #000000 !important;
    margin-top: 20px !important;
    margin-bottom: 15px !important;
}}

/* 4. WIDGET VISIBILITY - LARGE LABELS */
.stWidget label p {{
    color: #FFFFFF !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    text-shadow: 1px 1px 2px #000000;
}}

/* Radio & Select Labels */
[data-testid="stRadio"] label p {{
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
}}

/* File Uploader - PROMINENT & LARGE */
[data-testid="stFileUploader"] {{
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(200, 200, 200, 0.95) 100%) !important;
    padding: 30px !important;
    border-radius: 20px !important;
    border: 3px solid #FFD700 !important;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}}

[data-testid="stFileUploader"] section {{
    color: #000000 !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
}}

/* 5. ALERT BOXES - LARGE TEXT & VIBRANT */
.stSuccess {{
    background: linear-gradient(135deg, rgba(34, 139, 34, 0.98) 0%, rgba(46, 180, 46, 0.95) 100%) !important; 
    border: 3px solid #00FF00 !important;
    box-shadow: 0 8px 25px rgba(0, 255, 0, 0.3);
    padding: 20px !important;
    border-radius: 10px !important;
}}
.stSuccess p {{ 
    color: #FFFFFF !important; 
    font-weight: 900 !important; 
    font-size: 1.4rem !important;
    text-shadow: 1px 1px 2px #000000;
}}

.stInfo {{
    background: linear-gradient(135deg, rgba(0, 100, 200, 0.98) 0%, rgba(30, 144, 255, 0.95) 100%) !important; 
    border: 3px solid #00BFFF !important;
    box-shadow: 0 8px 25px rgba(0, 191, 255, 0.3);
    padding: 20px !important;
    border-radius: 10px !important;
}}
.stInfo p {{ 
    color: #FFFFFF !important; 
    font-weight: 900 !important; 
    font-size: 1.4rem !important;
    text-shadow: 1px 1px 2px #000000;
}}

.stWarning {{
    background: linear-gradient(135deg, rgba(220, 20, 20, 0.98) 0%, rgba(255, 50, 50, 0.95) 100%) !important; 
    border: 3px solid #FF4444 !important;
    box-shadow: 0 8px 25px rgba(255, 68, 68, 0.3);
    padding: 20px !important;
    border-radius: 10px !important;
}}
.stWarning p {{ 
    color: #FFFFFF !important; 
    font-weight: 900 !important; 
    font-size: 1.4rem !important;
    text-shadow: 1px 1px 2px #000000;
}}

.stError {{
    background: linear-gradient(135deg, rgba(220, 0, 0, 0.98) 0%, rgba(255, 0, 0, 0.95) 100%) !important;
    border: 3px solid #FF0000 !important;
    box-shadow: 0 8px 25px rgba(255, 0, 0, 0.4);
    padding: 20px !important;
    border-radius: 10px !important;
}}
.stError p {{
    color: #FFFFFF !important;
    font-weight: 900 !important;
    font-size: 1.4rem !important;
    text-shadow: 1px 1px 2px #000000;
}}

/* 6. METRICS - HUGE & PROMINENT */
[data-testid="stMetricValue"] {{ 
    color: #00FF00 !important;
    font-size: 4rem !important;
    font-weight: 900 !important;
    text-shadow: 2px 2px 8px #000000, 0px 0px 20px rgba(0, 255, 0, 0.5);
}}

[data-testid="stMetricLabel"] p {{
    color: #FFD700 !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    text-shadow: 1px 1px 3px #000000;
}}

/* 7. GENERAL TEXT - LARGER SIZE */
p, span, li {{
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
}}

h1 {{
    font-size: 3rem !important;
    font-weight: 900 !important;
    color: #FFD700 !important;
    text-shadow: 2px 2px 4px #000000 !important;
}}

h2 {{
    font-size: 2.5rem !important;
    font-weight: 900 !important;
    color: #FFD700 !important;
    text-shadow: 2px 2px 4px #000000 !important;
    padding: 15px 0 !important;
}}

h3 {{
    font-size: 2.2rem !important;
    font-weight: 900 !important;
    color: #FFD700 !important;
    text-shadow: 1px 1px 3px #000000 !important;
}}

/* 8. BUTTON STYLING - LARGE & ATTRACTIVE */
div.stButton > button {{
    font-weight: 900 !important;
    font-size: 1.2rem !important;
    text-transform: uppercase;
    border-radius: 12px !important;
    padding: 15px 30px !important;
    border: 2px solid #FFD700 !important;
    background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%) !important;
    color: #000000 !important;
    box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
    transition: all 0.3s ease;
}}

div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(255, 215, 0, 0.6);
}}

/* 9. CAPTION - LARGER TEXT */
.stCaption {{
    font-size: 1.1rem !important;
    color: #E0E0E0 !important;
    font-weight: 600 !important;
}}

/* 10. PROGRESS BAR - MORE VISIBLE */
.stProgress > div > div > div {{
    background-color: #00FF00 !important;
    height: 25px !important;
}}

/* 11. DIVIDER - STYLED */
hr {{
    border: 2px solid #FFD700 !important;
    margin: 30px 0 !important;
    box-shadow: 0 2px 10px rgba(255, 215, 0, 0.3);
}}

/* 12. DATAFRAME STYLING */
[data-testid="stDataFrame"] {{
    font-size: 1.1rem !important;
}}

.stDataFrame thead th {{
    font-size: 1.3rem !important;
    font-weight: 800 !important;
    background-color: #FFD700 !important;
    color: #000000 !important;
}}

.stDataFrame tbody td {{
    font-size: 1rem !important;
    font-weight: 600 !important;
}}
</style>

<div class="header-container">
    <img src="data:image/png;base64,{logo_left_base64}" class="logo-img">
    <div class="header-text">Tomato Variety Identification</div>
    <img src="data:image/png;base64,{logo_right_base64}" class="logo-img">
</div>
""",
    unsafe_allow_html=True,
)


# -------------------------------------------------
# 6. UI MAIN LAYOUT
# -------------------------------------------------
res_variety, res_colors = None, None

# Control Buttons
st.markdown("### 🎮 Controls")
col_view, col_download, col_help = st.columns(3)

with col_view:
    if st.button("👁️ VIEW ALL RECORDS", use_container_width=True): 
        st.session_state.show_predictions = True
with col_download:
    if st.button("📥 DOWNLOAD EXCEL", use_container_width=True):
        preds = fetch_all_predictions()
        if preds:
            excel = convert_predictions_to_excel(preds)
            if excel:
                st.download_button("✅ CONFIRM DOWNLOAD", excel, "tomato_records.xlsx", use_container_width=True)
        else:
            st.warning("No records to download or Database disconnected.")
with col_help:
    if st.button("❓ HELP & GUIDE", use_container_width=True):
        st.session_state.show_help = not st.session_state.show_help

st.divider()

# HELP SECTION
if st.session_state.show_help:
    st.markdown("## 📚 HOW TO USE THIS SYSTEM")
    
    help_col1, help_col2 = st.columns(2)
    
    with help_col1:
        st.markdown("### 🖼️ STEP 1: UPLOAD IMAGE")
        st.info("""
        **Two Methods Available:**
        
        📤 **Upload Image:**
        - Click "Drop or select photo"
        - Choose a JPG, PNG, or JPEG file from your device
        - Works best with clear, well-lit tomato photos
        
        📸 **Live Camera Scan:**
        - Use your device camera
        - Point at the tomato
        - Takes a real-time photo
        """)
        
        st.markdown("### 🎯 STEP 2: AI ANALYSIS")
        st.success("""
        The system will automatically:
        - 🤖 Identify tomato variety
        - 📊 Calculate AI confidence level
        - 🎨 Detect color pigments
        - 📈 Determine ripeness level
        """)
    
    with help_col2:
        st.markdown("### 💡 STEP 3: GET RECOMMENDATIONS")
        st.warning("""
        **Three Ripeness States:**
        
        🟢 **UNRIPE** (0-40%)
        - Best for shipping & transport
        - Longer shelf life
        
        🟠 **RIPE** (40-75%)
        - Perfect for retail markets
        - Ideal condition
        
        🔴 **OVERRIPE** (75-100%)
        - Needs immediate processing
        - Limited shelf time
        """)
        
        st.markdown("### 📦 STEP 4: SHELF LIFE INFO")
        st.info("""
        Get precise storage recommendations:
        - 🏠 Room Temperature days
        - ❄️ Refrigerated days
        - 🌡️ Ideal temperature range
        - 💬 Special care notes
        """)
    
    st.divider()
    
    st.markdown("### 🔍 UNDERSTANDING THE RESULTS")
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.markdown("#### 🎯 VARIETY LABEL")
        st.markdown("""
        **Tomato Types Recognized:**
        - Apollo Tomato
        - Atlas Tomato
        - Cherry Tomato
        - Diamante Tomato
        - Kinalabasa Tomato
        - Pear Tomato
        - Rio Grande Tomato
        - Roma Tomato
        """)
    
    with result_col2:
        st.markdown("#### 📊 CONFIDENCE SCORE")
        st.markdown("""
        **What it means:**
        - 90-100%: Very high confidence ✅
        - 70-89%: Good confidence 👍
        - 50-69%: Moderate confidence ⚠️
        - Below 50%: Low confidence ❌
        
        **Tip:** Ensure good lighting!
        """)
    
    with result_col3:
        st.markdown("#### 🌈 COLOR PIGMENTS")
        st.markdown("""
        **Dominant Colors:**
        - Red: Mature tomato
        - Orange: Semi-ripe
        - Yellow: Less ripe
        - Green: Unripe
        
        Shows the main color distribution in your image.
        """)
    
    st.divider()
    
    st.markdown("### 💾 DATA MANAGEMENT")
    st.markdown("""
    - ✅ All analyses are **automatically saved** to the database
    - 👁️ **VIEW ALL RECORDS** button shows analysis history
    - 📥 **DOWNLOAD EXCEL** exports all data for reports
    - 📊 Track trends over time with historical data
    """)
    
    st.divider()
    
    st.markdown("### ⚡ TIPS FOR BEST RESULTS")
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        st.markdown("""
        ✅ **DO:**
        - Use good natural lighting
        - Take clear, focused photos
        - Show the entire tomato
        - Use clean camera lens
        - Take multiple angles if unsure
        """)
    
    with tips_col2:
        st.markdown("""
        ❌ **DON'T:**
        - Use blurry or dark photos
        - Hide parts of the tomato
        - Use extreme angles
        - Crop too much
        - Use artificial filters
        """)
    
    if st.button("✕ Close Help"):
        st.session_state.show_help = False

st.divider()
col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

with col1:
    st.markdown("## 📷 IMAGE INPUT")
    option = st.radio("**Select Method:**", ("Upload Image", "Live Camera Scan"), horizontal=True)
    image_to_process = None
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("📁 Drop or select photo", type=["jpg","png","jpeg"])
        if uploaded_file: image_to_process = Image.open(uploaded_file)
    else:
        camera_photo = st.camera_input("📸 Scan Tomato Now")
        if camera_photo: image_to_process = Image.open(camera_photo)

    if image_to_process:
        st.markdown('<div style="border: 3px solid #FFD700; border-radius: 10px; padding: 5px; background: rgba(255,215,0,0.1);">', unsafe_allow_html=True)
        st.image(image_to_process, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        with st.spinner("🔍 Analyzing tomato variety..."):
            try:
                res_variety, res_colors = run_prediction(image_to_process)
                if res_variety:
                    res_variety["source"] = "Upload" if option == "Upload Image" else "Live Scan"
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")

with col2:
    st.markdown("## 📊 ANALYSIS RESULTS")
    if res_variety:
        v_label = res_variety.get("variety_label", "Unknown")
        st.success(f"✅ **VARIETY:** {v_label.replace('_',' ').title()}")
        st.metric("🎯 AI CONFIDENCE", res_variety.get('prediction_display'), delta=None)
        
        if v_label != "non_tomato":
            f_score = res_variety.get("fuzzy_ripeness", 0)
            rip_status = "🟢 UNRIPE" if f_score < 40 else "🟠 RIPE" if f_score < 75 else "🔴 OVERRIPE"
            st.markdown(f"### **RIPENESS STATUS:** {rip_status}")
            st.markdown(f"<h2 style='color: #00FF00; text-align: center;'>{f_score:.1f}%</h2>", unsafe_allow_html=True)
            st.progress(f_score / 100)
            
            if res_colors:
                st.markdown("#### 🌈 DOMINANT COLOR PIGMENTS")
                cols = st.columns(len(res_colors))
                for i, (lbl, val) in enumerate(res_colors.items()):
                    with cols[i]:
                        st.markdown(f'<div style="background:{val["color"]};height:60px;border-radius:10px;border:3px solid #FFD700;box-shadow: 0 4px 15px rgba(0,0,0,0.5);"></div>', unsafe_allow_html=True)
                        st.markdown(f"<h4 style='text-align: center; color: #FFD700;'>{lbl}</h4>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='text-align: center; color: #FFD700;'>⏳ Waiting for image input...</h3>", unsafe_allow_html=True)

with col3:
    st.markdown("## 💡 RECOMMENDATIONS")
    if res_variety and res_variety.get("variety_label") != "non_tomato":
        f_score = res_variety.get("fuzzy_ripeness", 0)
        if f_score < 40: 
            st.warning("🟢 LOGISTICS READY\nBest for shipping and transport.")
        elif f_score < 75: 
            st.success("🟠 MARKET READY\nPrime condition for retail.")
        else: 
            st.error("🔴 URGENT ACTION\nImmediate processing required!")
        
        rec = res_variety.get("recommendation")
        if isinstance(rec, dict):
            st.markdown(f"<h4 style='color: #E0E0E0;'>{rec.get('description', '')}</h4>", unsafe_allow_html=True)
            try:
                sl = adjust_shelf_life_for_ripeness(rec.get("shelf_life", {}), f_score)
                st.info(f"""
                📦 **SHELF LIFE ESTIMATE:**
                - 🏠 Room Temp: **{sl.get('room_temp_days')} days**
                - ❄️ Refrigerated: **{sl.get('refrigerated_days')} days**
                """)
            except:
                pass

            temp_info = rec.get("temperature_feasibility")
            if isinstance(temp_info, dict):
                ideal = temp_info.get("ideal_temp_c")
                note = temp_info.get("feasibility_note")
                if ideal:
                    st.markdown(f"### 🌡️ IDEAL TEMPERATURE: **{ideal[0]}°C - {ideal[1]}°C**")
                if note:
                    st.info(f"💬 {note}")
    else:
        st.markdown("<h3 style='text-align: center; color: #FFD700;'>📝 Recommendations will appear here after analysis.</h3>", unsafe_allow_html=True)

# -------------------------------------------------
# 7. AUTO SAVE TO DATABASE
# -------------------------------------------------
# Automatic save after successful prediction
if res_variety and supabase and res_variety.get("variety_label") != "Unknown":
    
    # Extract the recommendation field
    rec_data = res_variety.get("recommendation")
    
    # VALIDATION: Only auto-save if recommendation exists and is valid
    if rec_data is not None and str(rec_data).strip().lower() not in ["none", "", "null"]:
        try:
            # 1. Sanitize the prediction (Convert "89%" string to 0.89 float)
            raw_pred = res_variety.get("prediction", 0)
            if isinstance(raw_pred, str) and "%" in raw_pred:
                clean_pred = float(raw_pred.replace("%", "")) / 100.0
                display_pred = raw_pred
            else:
                clean_pred = float(raw_pred)
                display_pred = f"{int(clean_pred * 100)}%"

            # 2. Construct the final payload
            # Using a manual UUID prevents duplicates if re-analysis happens
            payload = {
                "id": str(uuid.uuid4()), 
                "variety_label": res_variety.get("variety_label"),
                "prediction": clean_pred,
                "prediction_display": display_pred,
                "status": res_variety.get("status"),
                "hsv_percent": convert_to_serializable(res_variety.get("hsv_percent")),
                "lab_score": convert_to_serializable(res_variety.get("lab_score")),
                "recommendation": convert_to_serializable(rec_data),
                "source": res_variety.get("source", "Upload"),
                "fuzzy_ripeness": convert_to_serializable(res_variety.get("fuzzy_ripeness"))
            }

            # 3. Execute the Supabase insert automatically
            supabase.table("tomato_logs").insert(payload).execute()
            
            st.success("✅ ANALYSIS SAVED TO DATABASE!")
            
        except Exception as e:
            st.warning(f"⚠️ Auto-save note: {e}")
# -------------------------------------------------
# 7. DATABASE RECORDS VIEW
# -------------------------------------------------
if st.session_state.get("show_predictions"):
    st.divider()
    st.markdown("## 📋 HISTORICAL ANALYSIS RECORDS")
    
    logs = fetch_all_predictions()
    if logs:
        st.markdown(f"### 📊 Total Records: **{len(logs)}**")
        st.dataframe(
            pd.DataFrame(logs), 
            use_container_width=True,
            height=500,
            hide_index=True
        )
        
        # Export button
        if st.button("📥 EXPORT DATA TO EXCEL"):
            excel_data = convert_predictions_to_excel(logs)
            if excel_data:
                st.download_button(
                    label="✅ DOWNLOAD RECORDS",
                    data=excel_data,
                    file_name="tomato_records_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    else:
        st.info("📊 No analysis records found yet. Upload an image to start!") 
