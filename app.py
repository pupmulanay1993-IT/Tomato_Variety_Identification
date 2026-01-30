import streamlit as st
from PIL import Image
import io
import numpy as np
import uuid
import tensorflow as tf
import av
import cv2
import base64
import json
import pandas as pd
from supabase import create_client, Client
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# Local utilities and functions
from utils import (
    clean_image,
    get_prediction,
    make_results,
    is_tomato_bouncer,
    detect_multi_colors,
    compute_color_scores,
)

# -------------------------------------------------
# 1. ADAPTIVE FUZZY LOGIC ENGINE DEFINITION
# -------------------------------------------------
# Antecedents (Inputs)
intensity = ctrl.Antecedent(np.arange(0, 101, 1), 'intensity')
accuracy = ctrl.Antecedent(np.arange(0, 101, 1), 'accuracy')
# Consequent (Output)
ripeness = ctrl.Consequent(np.arange(0, 101, 1), 'ripeness')

# Membership Functions
intensity.automf(3, names=['low', 'medium', 'high'])
accuracy.automf(3, names=['poor', 'average', 'good'])

ripeness['unripe'] = fuzz.trimf(ripeness.universe, [0, 0, 45])
ripeness['ripe'] = fuzz.trimf(ripeness.universe, [35, 65, 85])
ripeness['overripe'] = fuzz.trimf(ripeness.universe, [75, 100, 100])

# RULES: Strict logic for ripeness determination
rule1 = ctrl.Rule(intensity['low'], ripeness['unripe'])
rule2 = ctrl.Rule(accuracy['poor'], ripeness['unripe']) 
rule3 = ctrl.Rule(intensity['medium'] & accuracy['good'], ripeness['ripe'])
rule4 = ctrl.Rule(intensity['high'] & accuracy['good'], ripeness['ripe'])
rule5 = ctrl.Rule(intensity['high'] & accuracy['average'], ripeness['overripe'])

ripeness_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
ripeness_sim = ctrl.ControlSystemSimulation(ripeness_ctrl)

# -------------------------------------------------
# 2. PERFORMANCE CACHING & DB CONNECTION
# -------------------------------------------------
@st.cache_resource
def load_tomato_model():
    try:
        return tf.keras.models.load_model("tomato_model.keras", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def init_supabase() -> Client:
    try:
        return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except:
        return None

@st.cache_data
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

supabase = init_supabase()
model = load_tomato_model()
models = [model] if model else []

# Load Class Indices
try:
    with open("class_indices.json", "r") as f:
        class_mapping = json.load(f)
    idx_to_label = {int(v): k for k, v in class_mapping.items()}
except:
    idx_to_label = {0: "apollo_tomato", 1: "atlas_tomato", 2: "cherry_tomato", 3: "diamante_tomato", 
                    4: "kinalabasa_tomato", 5: "non_tomato", 6: "pear_tomato", 7: "rio_grande_tomato", 8: "roma_tomato"}

def convert_to_serializable(obj):
    """
    Converts NumPy types and other non-standard objects 
    into standard Python types for JSON compatibility.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def fetch_all_predictions():
    """Fetch all prediction records from Supabase."""
    if not supabase:
        return None
    try:
        response = supabase.table("tomato_logs").select("*").execute()
        return response.data if response.data else []
    except Exception as e:
        st.error(f"Failed to fetch predictions: {e}")
        return None

def convert_predictions_to_excel(predictions):
    """Convert predictions to Excel format."""
    if not predictions:
        return None
    
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
# 3.1. VARIETY-AWARE PREDICTION LOGIC
# -------------------------------------------------
def run_prediction(pil_image):
    # STEP A: Identification (CNN)
    img_clean = clean_image(pil_image, target_size=(224, 224))
    probs, idx, conf = get_prediction(model, img_clean)
    variety_label = idx_to_label.get(idx, "Unknown")

    # STEP B: Color Analysis (Variety-Aware)
    hsv_p, lab_s = compute_color_scores(pil_image, variety_label=variety_label)

    # STEP C: Fuzzy Logic Computation
    ripeness_sim.input['intensity'] = np.clip(hsv_p, 0, 100)
    ripeness_sim.input['accuracy'] = np.clip(lab_s * 100 if lab_s <= 1 else lab_s, 0, 100)
    
    try:
        ripeness_sim.compute()
        fuzzy_score = ripeness_sim.output['ripeness']
    except:
        fuzzy_score = 0

    # STEP D: Format Final Results
    result = make_results(probs, idx, conf, class_indices_path="class_indices.json")
    result.update({
        "fuzzy_ripeness": float(fuzzy_score),
        "hsv_score": hsv_p,
        "lab_score": lab_s
    })
    
    # Color clustering for UI
    img_rgb = np.array(pil_image.convert("RGB"))
    res_colors = detect_multi_colors(img_rgb, k=4)
    
    return result, res_colors

# -------------------------------------------------
# STYLING & HEADER (COMPLETE & MOBILE-OPTIMIZED)
# -------------------------------------------------
st.set_page_config(page_title="Tomato Variety Identification", layout="wide", page_icon="favicon.png")

background_base64 = get_base64_of_bin_file("background.jpg")
logo_left_base64 = get_base64_of_bin_file("PUP Mulanay left.png")
logo_right_base64 = get_base64_of_bin_file("PUP Mulanay right.png")

st.markdown(
    f"""
<style>
/* 1. BASE APP & BACKGROUND */
.stApp {{
    background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                      url("data:image/jpg;base64,{background_base64}");
    background-size: cover;
    background-position: center;
    color: white !important;
    font-weight: 600;
}}

/* 2. RESPONSIVE HEADER (LOGO - TEXT - LOGO) */
.header-container {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0px;
    margin-bottom: 20px;
}}

.header-text {{
    font-size: clamp(1.2rem, 5vw, 2rem); /* Dynamic para sa selpon */
    font-weight: bold;
    color: #FFD700 !important;
    text-align: center;
    flex-grow: 1;
    line-height: 1.2;
}}

.logo-img {{
    width: clamp(45px, 10vw, 65px); /* Responsive logos */
    height: auto;
}}

/* 3. RADIO & UPLOADER VISIBILITY */
.stRadio label, .stRadio div, .stRadio span {{ color: white !important; }}
.stFileUploader {{ 
    background-color: rgba(255,255,255,0.9); 
    border-radius: 10px; 
    padding: 10px; 
    color: #000000 !important;
}}
.stFileUploader p, .stFileUploader span, .uploadedFile {{ color: #000000 !important; }}

/* 4. ALERTS & MESSAGES (YOUR ORIGINAL COLORS) */
.stSuccess {{
    color: white !important;
    background-color: rgba(0, 100, 0, 0.8) !important;
    padding: 15px !important;
    border-radius: 5px !important;
}}
.stSuccess p, .stSuccess strong {{ color: #FF6600 !important; font-size: 18px !important; font-weight: bold !important; }}

.stInfo {{
    color: white !important;
    background-color: rgba(0, 50, 100, 0.2) !important;
    padding: 15px !important;
    border-radius: 5px !important;
}}
.stInfo p, .stInfo strong {{ color: #FF6600 !important; font-size: 16px !important; font-weight: bold !important; }}

.stWarning {{
    color: white !important;
    background-color: rgba(100, 50, 0, 0.5) !important;
    padding: 15px !important;
    border-radius: 5px !important;
}}
.stWarning p, .stWarning strong {{ color: #FFFFFF !important; font-size: 16px !important; font-weight: bold !important; }}

/* 5. BUTTONS & METRICS */
div.stButton > button {{
    font-size: 12px !important;
    padding: 5px 10px !important;
    border-radius: 8px !important;
    text-transform: uppercase;
    font-weight: bold !important;
}}
[data-testid="stMetricValue"] {{ color: #FFFFFF !important; font-weight: bold !important; }}
h1, h2, h3, h4, h5, h6 {{ color: #FFFFFF !important; }}

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
# 3.2. PREDICTION HELPER (Optimized for Variety-Aware Fuzzy Logic)
# -------------------------------------------------
def _get_model_input_size(model, fallback=(224, 224)):
    try:
        if hasattr(model, "inputs") and model.inputs:
            ishape = model.inputs[0].shape
            h = int(ishape[1]) if ishape[1] else None
            w = int(ishape[2]) if ishape[2] else None
            if h and w: return (h, w)
        if hasattr(model, "input_shape") and model.input_shape:
            ishape = model.input_shape
            h = int(ishape[1]) if ishape[1] else None
            w = int(ishape[2]) if ishape[2] else None
            if h and w: return (h, w)
    except: pass
    return fallback

def run_prediction(pil_image):
    # STEP A: Identification (CNN Inference)
    # Kailangan muna malaman ang variety bago ang kulay
    img_rgb = np.array(pil_image.convert("RGB"))
    preds_list = []
    
    # Suporta para sa multiple models (Ensemble)
    for m in models:
        h, w = _get_model_input_size(m)
        img_clean = clean_image(pil_image, target_size=(h, w))
        preds, indices, confs = get_prediction(m, img_clean)
        preds_list.append(preds)

    if not preds_list:
        return None, None

    avg_preds = np.mean(preds_list, axis=0)
    idx = int(np.argmax(avg_preds))
    conf = float(np.max(avg_preds))
    detected_variety = idx_to_label.get(idx, "Unknown")

    # STEP B: Variety-Aware Color Scoring
    # Dito ipinapasa ang variety_label para mag-adjust ang logic sa Yellow o Red
    hsv_percent, lab_score = compute_color_scores(pil_image, variety_label=detected_variety)
    tomato_like = is_tomato_bouncer(pil_image)

    # STEP C: Fuzzy Computation
    try:
        ripeness_sim.input['intensity'] = hsv_percent
        # I-normalize ang lab_score kung 0-1 range siya
        ripeness_sim.input['accuracy'] = lab_score * 100 if lab_score <= 1.0 else lab_score
        ripeness_sim.compute()
        fuzzy_score = ripeness_sim.output['ripeness']
    except:
        fuzzy_score = 0

    # STEP D: Result Formatting
    result = make_results(avg_preds, idx, conf, class_indices_path="class_indices.json")
    result.update({
        "variety_label": detected_variety,
        "prediction": float(conf),  # Store as float for database compatibility
        "prediction_display": f"{int(conf * 100)}%",  # Formatted version for UI
        "hsv_percent": float(hsv_percent),
        "lab_score": float(lab_score),
        "fuzzy_ripeness": float(fuzzy_score),
        "status": "Valid" if tomato_like else "Low Color Match"
    })

    # Color detection for visual representation
    color_map = {"Red Tomato": "#FF0000", "Orange Tomato": "#FF7F00", "Yellow Tomato": "#FFFF00", "Green Tomato": "#00FF00", "Other": "#999999"}
    res_colors = detect_multi_colors(img_rgb, k=4, color_map=color_map)

    return result, res_colors

# -------------------------------------------------
# 4. VIDEO TRANSFORMER FOR LIVE CAMERA
# -------------------------------------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.latest_frame = None
    
    def transform(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_frame = cv2.flip(img, 1)
        return self.latest_frame

# -------------------------------------------------
# 5. UI LAYOUT & DISPLAY (STRICT HORIZONTAL)
# -------------------------------------------------
btn_col1, btn_col2 = st.columns(2)

with btn_col1:
    # Orange "View All" - compact version
    st.markdown("""
        <style>
        div[data-testid="column"]:nth-of-type(1) button {
            background-color: #FF6600 !important;
            color: white !important;
            border: 1px solid #FFD700 !important;
            height: 35px !important; /* Pinababa ang height */
            font-size: 10px !important; /* Mas maliit na text para hindi mag-wrap */
            margin-bottom: 0px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    if st.button("👁️ VIEW ALL", use_container_width=True, key="btn_view"):
        st.session_state.show_predictions = True

with btn_col2:
    # Blue "Download" - compact version
    st.markdown("""
        <style>
        div[data-testid="column"]:nth-of-type(2) button {
            background-color: #1E90FF !important;
            color: white !important;
            border: 1px solid #FFD700 !important;
            height: 35px !important;
            font-size: 10px !important;
            margin-bottom: 0px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    if st.button("📥 DOWNLOAD", use_container_width=True, key="btn_download"):
        predictions = fetch_all_predictions()
        if predictions:
            excel_data = convert_predictions_to_excel(predictions)
            if excel_data:
                st.download_button(
                    label="OK",
                    data=excel_data,
                    file_name="tomato_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="excel_trigger"
                )

st.markdown("<div style='margin-bottom: -15px;'></div>", unsafe_allow_html=True) # Para itaas pa lalo ang divider
st.divider()

# Para siguradong angat ang col1, col2, col3
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("📷 Image Input")
    option = st.radio("Choose Method:", ("Upload Image", "Live Camera Scan"), horizontal=True)

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Drop tomato photo here", type=["jpg","png","jpeg"])
        if uploaded_file:
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, use_container_width=True, caption="Target Image")
            
            with st.spinner("🔍 AI is analyzing..."):
                res_variety, res_colors = run_prediction(image_to_process)
                res_variety["source"] = "Upload"
                
                # Auto-sync to Supabase
                if supabase and res_variety.get("variety_label") != "Unknown":
                    try:
                        save_data = res_variety.copy()
                        if "recommendation" in save_data and isinstance(save_data["recommendation"], dict):
                            save_data["recommendation"] = json.dumps(save_data["recommendation"])
                        
                        # Siguraduhin ang unique ID
                        save_data["id"] = str(uuid.uuid4())
                        
                        supabase.table("tomato_logs").insert([save_data]).execute()
                        st.caption("✅ Saved to database.")
                    except Exception as e:
                        st.error(f"Sync failed: {e}")

    else:
        # FAIL-SAFE CAMERA: Gamit ang st.camera_input
        st.info("💡 Tip: Click 'Take Photo' to scan the tomato variety.")
        
        # Bubuksan nito ang native camera app ng cellphone mo
        camera_photo = st.camera_input("Scan Tomato")

        if camera_photo:
            # I-convert ang photo para ma-process ng model
            image_to_process = Image.open(camera_photo)
            
            # I-display ang litrato sa interface
            st.image(image_to_process, use_container_width=True, caption="Captured Image")
            
            with st.spinner("🔍 Analyzing tomato variety..."):
                res_variety, res_colors = run_prediction(image_to_process)
                res_variety["source"] = "Live Scan"
                
                # Auto-sync to Supabase
                if supabase and res_variety.get("variety_label") != "Unknown":
                    try:
                        save_data = res_variety.copy()
                        if "recommendation" in save_data and isinstance(save_data["recommendation"], dict):
                            save_data["recommendation"] = json.dumps(save_data["recommendation"])
                        
                        # Metadata/ID
                        save_data["id"] = str(uuid.uuid4())
                        
                        supabase.table("tomato_logs").insert([save_data]).execute()
                        st.success("✅ Analysis successful and saved to history!")
                    except Exception as e:
                        st.error(f"Sync failed: {e}")
                        
with col2:
    st.subheader("📊 Processing")
    if res_variety:
        v_name = res_variety["variety_label"].replace("_"," ").title()
        st.success(f"**Variety:** {v_name}")
        st.metric("AI Confidence", res_variety.get('prediction_display', f"{res_variety.get('prediction', 0):.1%}"))
        
        st.divider()
        
        # Ripeness Gauge using Fuzzy Logic
        f_score = res_variety.get("fuzzy_ripeness", 0)
        if f_score < 40:
            rip_status, rip_color = "Unripe (Hilaw)", "green"
        elif f_score < 75:
            rip_status, rip_color = "Ripe (Hinog)", "orange"
        else:
            rip_status, rip_color = "Overripe (Lanta)", "red"
            
        st.markdown(f"**Ripeness Status:** :{rip_color}[{rip_status}]")
        st.metric("Maturity Level", f"{f_score:.1f}%")
        st.progress(f_score / 100)
        
        # Color Cluster UI
        if res_colors:
            st.subheader("Dominant Pigments")
            color_cols = st.columns(len(res_colors))
            for i, (lbl, val) in enumerate(res_colors.items()):
                with color_cols[i]:
                    st.markdown(f"""
                    <div style="
                        background-color: {val['color']};
                        height: 40px;
                        border-radius: 8px;
                        border: 2px solid white;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        margin: 5px 0;
                    ">
                        <span style="color: white; font-weight: bold; font-size: 10px; text-shadow: 1px 1px 2px black;">
                            {lbl.split()[0]}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Analysis Metrics Box
            st.markdown("---")
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid #FFD700;
            ">
                <h4 style="color: white; margin: 0; text-transform: uppercase;">📊 Analysis Metrics</h4>
                <p style="color: #E0E0E0; margin: 5px 0; font-size: 12px;">
                    <b>HSV Score:</b> """ + f"{res_variety.get('hsv_score', 0):.1f}%" + """
                    | <b>Lab Score:</b> """ + f"{res_variety.get('lab_score', 0):.3f}" + """
                </p>
            </div>
            """, unsafe_allow_html=True)

with col3:
    st.subheader("💡 Recommendations")
    if res_variety:
        f_score = res_variety.get("fuzzy_ripeness", 0)
        
        # Insight based on Maturity
        if f_score < 40:
            st.warning("🟢 **Logistics:** High durability. Best for long-distance shipping.")
        elif f_score < 75:
            st.success("🟠 **Market:** Prime condition for retail and grocery sales.")
        else:
            st.error("🔴 **Urgent:** Short shelf-life. Immediate processing (sauce/paste) recommended.")
        
        # Variety Deep-Dive
        rec = res_variety.get("recommendation")
        if isinstance(rec, dict):
            st.subheader("Description")
            st.write(rec.get("description"))
            st.subheader("Lifespan")
            st.write(rec.get("plant_lifespan"))
            st.markdown(f"**{v_name} Characteristics:**")
            st.caption(rec.get('description'))
            
            # Shelf Life Display
            sl = rec.get("shelf_life", {})
            st.markdown(f"""
                <div style="display:flex; gap:10px;">
                    <div style="flex:1; background:#FFF3E0; padding:10px; border-radius:8px; text-align:center; border:1px solid #FFB74D;">
                        <small>🏠 Room</small><br><b>{sl.get('room_temp_days')} Days</b>
                    </div>
                    <div style="flex:1; background:#E3F2FD; padding:10px; border-radius:8px; text-align:center; border:1px solid #64B5F6;">
                        <small>❄️ Fridge</small><br><b>{sl.get('refrigerated_days')} Days</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Climate Info
            if "temperature_feasibility" in rec:
                tf = rec["temperature_feasibility"]
                st.info(f"🌡️ **Ideal Temp:** {tf['ideal_temp_c'][0]}°C - {tf['ideal_temp_c'][1]}°C")

# -------------------------------------------------
# 6. SUPABASE SYNC
# -------------------------------------------------
# Check if the analysis result exists and is valid
if res_variety and supabase and res_variety.get("variety_label") != "Unknown":
    
    # Extract the recommendation field
    rec_data = res_variety.get("recommendation")
    
    # UI Button to prevent automatic/duplicate triggers
    if st.button("Save Analysis to Database"):
        
        # VALIDATION: Block the insert if recommendation is missing/empty
        if rec_data is None or str(rec_data).strip().lower() in ["none", "", "null"]:
            st.error("Validation Failed: Recommendation is still empty. Please wait for the analysis to complete.")
        else:
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
                # Using a manual UUID prevents duplicates if the button is clicked twice
                payload = {
                    "id": str(uuid.uuid4()), 
                    "variety_label": res_variety.get("variety_label"),
                    "prediction": clean_pred,
                    "prediction_display": display_pred,
                    "status": res_variety.get("status"),
                    "hsv_percent": convert_to_serializable(res_variety.get("hsv_percent")),
                    "lab_score": convert_to_serializable(res_variety.get("lab_score")),
                    "recommendation": convert_to_serializable(rec_data), # Guaranteed to have content here
                    "source": res_variety.get("source", "Upload"),
                    "fuzzy_ripeness": convert_to_serializable(res_variety.get("fuzzy_ripeness"))
                }

                # 3. Execute the Supabase insert
                supabase.table("tomato_logs").insert(payload).execute()
                
                st.success("✅ Record saved successfully with full recommendation data!")
                
            except Exception as e:
                st.error(f"Database Error: {e}")
# -------------------------------------------------
# 7. DATABASE RECORDS VIEW
# -------------------------------------------------
if st.session_state.get("show_predictions"):
    st.divider()
    st.header("📋 Historical Analysis Records")
    
    logs = fetch_all_predictions()
    if logs:
        st.markdown(f"**Total Records: {len(logs)}**")
        st.dataframe(
            pd.DataFrame(logs), 
            use_container_width=True,
            height=500,
            hide_index=True
        )
        
        # Export button
        if st.button("📥 Export Visible Data to Excel"):
            excel_data = convert_predictions_to_excel(logs)
            if excel_data:
                st.download_button(
                    label="Download as Excel",
                    data=excel_data,
                    file_name="tomato_records_export.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    else:
        st.info("📊 No analysis records found yet. Upload an image to start!") 
