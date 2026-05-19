import streamlit as st
import os

# -------------------------------------------------
# 1. PAGE CONFIG (DAPAT ITO ANG PINAKA-UNA AT SINGLETON!)
# -------------------------------------------------
st.set_page_config(page_title="Tomato Variety Identification", layout="wide", page_icon="favicon.png")

# QUICK INITIALIZATION PARA HINDI MAG-HANG ANG STATE TRACKER
if "show_predictions" not in st.session_state:
    st.session_state.show_predictions = False

from PIL import Image
import io
import numpy as np
import uuid
import base64
import json
import pandas as pd
from supabase import create_client, Client

# BAGONG LIBRARY PARA SA TFLITE (MAGAAN AT MABILIS)
import tflite_runtime.interpreter as tflite

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
# 2. PERFORMANCE CACHING & DB CONNECTION
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

# Tanging magagaang initializers lang ang maiiwan dito sa global scope
supabase = init_supabase()

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
# 3. PREDICTION & FUZZY SYSTEM FUNCTION
# -------------------------------------------------
def _get_model_input_size(interpreter, fallback=(224, 224)):
    """
    Kumukuha ng input dimensions (height, width) mula sa TFLite interpreter details.
    """
    try:
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape'] # Format: [Batch, Height, Width, Channels]
        h, w = int(input_shape[1]), int(input_shape[2])
        if h and w: return (h, w)
    except: pass
    return fallback

def run_prediction(pil_image):
    # LAZY IMPORT AT LAZY MODEL LOADING
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    
    interpreter = load_tomato_model()
    if not interpreter:
        st.error("No TFLite model loaded to perform prediction.")
        return None, None

    img_rgb = np.array(pil_image.convert("RGB"))
    
    # Kunin ang tamang sukat (halimbawa: 224x224) base sa TFLite interpreter
    h, w = _get_model_input_size(interpreter)
    img_clean = clean_image(pil_image, target_size=(h, w))
    
    # UPDATE: Dahil utils.py ay baka umaasa pa sa lumang Keras format, 
    # ipinapasa natin ang interpreter sa get_prediction function mo.
    # Tiyakin na ang utils.py mo ay updated din para mag-invoke via interpreter.
    preds, indices, confs = get_prediction(interpreter, img_clean)

    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    detected_variety = idx_to_label.get(idx, "Unknown")

    hsv_percent, lab_score = compute_color_scores(pil_image, variety_label=detected_variety)
    tomato_like = is_tomato_bouncer(pil_image)

    # SECURED LOCAL FUZZY LOGIC COMPUTATION
    try:
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

        ripeness_sim.input['intensity'] = hsv_percent
        ripeness_sim.input['accuracy'] = lab_score * 100 if lab_score <= 1.0 else lab_score
        ripeness_sim.compute()
        fuzzy_score = ripeness_sim.output['ripeness']
    except Exception as e:
        fuzzy_score = 0

    result = make_results(preds, idx, conf, class_indices_path="class_indices.json")
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
# 4. STYLING & HEADER (SAFE RENDERING)
# -------------------------------------------------
background_base64 = get_base64_of_bin_file("background.jpg")
logo_left_base64 = get_base64_of_bin_file("PUP Mulanay left.png")
logo_right_base64 = get_base64_of_bin_file("PUP Mulanay right.png")

style_bg = f'url("data:image/jpg;base64,{background_base64}")' if background_base64 else '#121212'

st.markdown(f"""
<style>
.stApp {{ 
    background-image: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), {style_bg}; 
    background-size: cover; 
    background-position: center; 
    color: #FFFFFF !important; 
}}
.header-container {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 0px; margin-bottom: 30px; }}
.logo-img {{ width: clamp(50px, 12vw, 80px); height: auto; }}
.header-text {{ font-size: clamp(1.5rem, 5vw, 2.5rem); font-weight: 800; color: #FFD700 !important; text-shadow: 2px 2px 4px #000000; text-align: center; flex-grow: 1; }}
p, span, li, h1, h2, h3, label {{ color: #FFFFFF !important; }}
</style>
<div class="header-container">
    {"<img src='data:image/png;base64," + logo_left_base64 + "' class='logo-img'>" if logo_left_base64 else "<div></div>"}
    <div class="header-text">Tomato Variety Identification</div>
    {"<img src='data:image/png;base64," + logo_right_base64 + "' class='logo-img'>" if logo_right_base64 else "<div></div>"}
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 5. UI MAIN LAYOUT
# -------------------------------------------------
res_variety, res_colors = None, None
col_view, col_download = st.columns(2)

with col_view:
    if st.button("👁️ VIEW ALL", use_container_width=True): 
        st.session_state.show_predictions = True
with col_download:
    if st.button("📥 DOWNLOAD", use_container_width=True):
        preds = fetch_all_predictions()
        if preds:
            excel = convert_predictions_to_excel(preds)
            if excel:
                st.download_button("CONFIRM DOWNLOAD", excel, "tomato_records.xlsx", use_container_width=True)
        else:
            st.warning("No records to download or Database disconnected.")

st.divider()
col1, col2, col3 = st.columns([1, 1, 1], gap="small")

with col1:
    st.subheader("📷 Image Input")
    option = st.radio("Method:", ("Upload Image", "Live Camera Scan"), horizontal=True)
    image_to_process = None
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Drop photo", type=["jpg","png","jpeg"])
        if uploaded_file: image_to_process = Image.open(uploaded_file)
    else:
        camera_photo = st.camera_input("Scan Tomato")
        if camera_photo: image_to_process = Image.open(camera_photo)

    if image_to_process:
        st.image(image_to_process, use_container_width=True)
        with st.spinner("🔍 Analyzing..."):
            try:
                res_variety, res_colors = run_prediction(image_to_process)
                if res_variety:
                    res_variety["source"] = "Upload" if option == "Upload Image" else "Live Scan"
            except Exception as e:
                st.error(f"Analysis failed: {e}")

with col2:
    st.subheader("📊 Processing")
    if res_variety:
        v_label = res_variety.get("variety_label", "Unknown")
        st.success(f"**Variety:** {v_label.replace('_',' ').title()}")
        st.metric("AI Confidence", res_variety.get('prediction_display'))
        
        if v_label != "non_tomato":
            f_score = res_variety.get("fuzzy_ripeness", 0)
            rip_status = "Unripe" if f_score < 40 else "Ripe" if f_score < 75 else "Overripe"
            st.markdown(f"**Ripeness:** {rip_status}")
            st.progress(f_score / 100)
            
            if res_colors:
                st.subheader("Dominant Pigments")
                cols = st.columns(len(res_colors))
                for i, (lbl, val) in enumerate(res_colors.items()):
                    cols[i].markdown(f'<div style="background:{val["color"]};height:30px;border-radius:5px;border:1px solid white;"></div>', unsafe_allow_html=True)
    else:
        st.caption("Waiting for image input...")

with col3:
    st.subheader("💡 Recommendations")
    if res_variety and res_variety.get("variety_label") != "non_tomato":
        f_score = res_variety.get("fuzzy_ripeness", 0)
        if f_score < 40: st.warning("🟢 Logistics: Best for shipping.")
        elif f_score < 75: st.success("🟠 Market: Prime for retail.")
        else: st.error("🔴 Urgent: Immediate processing needed.")
        
        rec = res_variety.get("recommendation")
        if isinstance(rec, dict):
            st.caption(rec.get("description", ""))
            try:
                sl = adjust_shelf_life_for_ripeness(rec.get("shelf_life", {}), f_score)
                st.info(f"🏠 Room: {sl.get('room_temp_days')} days | ❄️ Fridge: {sl.get('refrigerated_days')} days")
            except:
                pass
    else:
        st.caption("Recommendations will appear here after analysis.")

# -------------------------------------------------
# 6. SAVE TO DATABASE
# -------------------------------------------------
if res_variety and supabase and res_variety.get("variety_label") != "Unknown":
    if st.button("Save Analysis to Database"):
        try:
            payload = {
                "id": str(uuid.uuid4()), 
                "variety_label": res_variety.get("variety_label"),
                "prediction": res_variety.get("prediction"),
                "status": res_variety.get("status"),
                "hsv_percent": convert_to_serializable(res_variety.get("hsv_percent")),
                "lab_score": convert_to_serializable(res_variety.get("lab_score")),
                "recommendation": convert_to_serializable(res_variety.get("recommendation")),
                "source": res_variety.get("source"),
                "fuzzy_ripeness": convert_to_serializable(res_variety.get("fuzzy_ripeness"))
            }
            supabase.table("tomato_logs").insert(payload).execute()
            st.success("✅ Saved successfully!")
        except Exception as e: 
            st.error(f"Database save error: {e}")

if st.session_state.get("show_predictions"):
    st.divider()
    logs = fetch_all_predictions()
    if logs: 
        st.dataframe(pd.DataFrame(logs), use_container_width=True)
