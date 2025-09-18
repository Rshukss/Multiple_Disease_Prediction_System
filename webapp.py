import pickle
from pathlib import Path
from typing import List, Any, Dict, Tuple
import streamlit as st
from streamlit_option_menu import option_menu

APP_TITLE = "Multiple Disease Prediction System"
MODEL_DIR = Path("Saved Models")
DISCLAIMER = (
    "This application is for educational/demo purposes only and is NOT a "
    "substitute for professional medical diagnosis or advice."
)

# Feature spec:
# type: "float" | "int" | "select"
# For select provide "choices"
DISEASES: Dict[str, Dict[str, Any]] = {
    "Diabetes Prediction": {
        "model_file": "diabetes_model.sav",
        "features": [
            {"label": "Pregnancies", "type": "int"},
            {"label": "Glucose", "type": "int"},
            {"label": "Insulin", "type": "int"},
            {"label": "BMI", "type": "float"},
            {"label": "Diabetes Pedigree Function", "type": "float"},
            {"label": "Age", "type": "int"},
        ],
        "positive_label": "Diabetes Positive",
        "negative_label": "Diabetes Negative",
        "columns_per_row": 3,
        "extra_help": """
**Feature Notes**
Pregnancies, Glucose (OGTT), Insulin (2hr), BMI (kg/m²), Pedigree (family history), Age (years).
""",
    },
    "Heart Disease Prediction": {
        "model_file": "heart_model.sav",
        "features": [
            {"label": "Age", "type": "int"},
            {"label": "sex", "type": "select", "choices": [0, 1]}, 
            {"label": "Chest Pain", "type": "select", "choices": [0,1,2,3]},
            {"label": "Resting Blood Pressure", "type": "int"},
            {"label": "Serum Cholesterol (mg/dl)", "type": "int"},
            {"label": "Fasting Blood Sugar", "type": "select", "choices": [0,1]},
            {"label": "Resting ECG", "type": "select", "choices": [0,1,2]},
            {"label": "Max Heart Rate", "type": "int"},
            {"label": "Exercise Induced Angina", "type": "select", "choices": [0,1]},
            {"label": "ST Depression (oldpeak)", "type": "float"},
            {"label": "Slope", "type": "select", "choices": [0,1,2]},
            {"label": "Flourosopy Vessels", "type": "select", "choices": [0,1,2,3]},
            {"label": "thal", "type": "select", "choices": [0, 1, 2, 3]},
        ],
        "positive_label": "Heart Disease Positive",
        "negative_label": "Heart Disease Negative",
        "columns_per_row": 4,
        "extra_help": """
Chest Pain: 0=Typical 1=Atypical 2=Non-anginal 3=Asymptomatic  
FBS: 1 >120 mg/dl else 0  
Rest ECG: 0=Normal 1=ST-T abn 2=LVH  
Angina: 1=Yes 0=No (dataset dependent)  
Slope: 0=Upsloping 1=Flat 2=Downsloping  
Flourosopy Vessels: 0-3
Sex: 0=Female 1=Male
Thal: 0=Unknown, 1=Normal, 2=Fixed Defect, 3=Reversible Defect
""",
    },
    "Parkinson's Disease Prediction": {
        "model_file": "parkinsons.sav",
        "features": [
            {"label": "MDVP:Fo(Hz)", "type": "float"},
            {"label": "MDVP:Fhi(Hz)", "type": "float"},
            {"label": "MDVP:Flo(Hz)", "type": "float"},
            {"label": "MDVP:Jitter(%)", "type": "float"},
            {"label": "MDVP:Jitter(Abs)", "type": "float"},
            {"label": "MDVP:RAP", "type": "float"},
            {"label": "MDVP:PPQ", "type": "float"},
            {"label": "MDVP:Shimmer", "type": "float"},
            {"label": "MDVP:Shimmer(dB)", "type": "float"},
            {"label": "MDVP:APQ", "type": "float"},
            {"label": "NHR", "type": "float"},
            {"label": "HNR", "type": "float"},
            {"label": "DFA", "type": "float"},
            {"label": "spread1", "type": "float"},
            {"label": "spread2", "type": "float"},
            {"label": "PPE", "type": "float"},
        ],
        "positive_label": "Parkinson's Positive",
        "negative_label": "Parkinson's Negative",
        "columns_per_row": 5,
        "extra_help": """
Acoustic perturbation and non-linear phonation metrics (jitter, shimmer, RAP, PPQ, APQ, NHR, HNR, PPE, RPDE, spread measures, DFA).
""",
    },
    "Breast Cancer Prediction": {
        "model_file": "breastCancer.sav",
        "features": [
            {"label": "radius_mean", "type": "float"},
            {"label": "texture_mean", "type": "float"},
            {"label": "perimeter_mean", "type": "float"},
            {"label": "area_mean", "type": "float"},
            {"label": "smoothness_mean", "type": "float"},
            {"label": "compactness_mean", "type": "float"},
            {"label": "concavity_mean", "type": "float"},
            {"label": "concave points_mean", "type": "float"},
            {"label": "symmetry_mean", "type": "float"},
            {"label": "radius_se", "type": "float"},
            {"label": "perimeter_se", "type": "float"},
            {"label": "area_se", "type": "float"},
            {"label": "concave points_se", "type": "float"},
            {"label": "radius_worst", "type": "float"},
            {"label": "texture_worst", "type": "float"},
            {"label": "perimeter_worst", "type": "float"},
            {"label": "area_worst", "type": "float"},
            {"label": "smoothness_worst", "type": "float"},
            {"label": "compactness_worst", "type": "float"},
            {"label": "concavity_worst", "type": "float"},
            {"label": "concave points_worst", "type": "float"},
            {"label": "symmetry_worst", "type": "float"},
            {"label": "fractal_dimension_worst", "type": "float"},
        ],
        "positive_label": "Breast Cancer Positive",
        "negative_label": "Breast Cancer Negative",
        "columns_per_row": 5,
        "extra_help": """
Shape, texture, and size descriptors (Mean / SE / Worst subsets). Higher irregularity often correlates with malignancy.
""",
    },
}

@st.cache_resource(show_spinner=False)
def load_model(file_name: str):
    path = MODEL_DIR / file_name
    with path.open("rb") as f:
        return pickle.load(f)

def get_prediction(model, features: List[float]) -> Tuple[Any, Any]:
    pred = model.predict([features])[0]
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba([features])[0]
        except Exception:
            prob = None
    return pred, prob

def render_feature_inputs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cols_per_row = cfg.get("columns_per_row", 3)
    values: Dict[str, Any] = {}
    features = cfg["features"]
    for i, feat in enumerate(features):
        if i % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        col = cols[i % cols_per_row]
        label = feat["label"]
        ftype = feat["type"]
        key = f"{cfg['model_file']}_{label}"
        if ftype == "select":
            values[label] = col.selectbox(label, feat["choices"], key=key)
        else:
            values[label] = col.text_input(label, key=key, placeholder="Enter value")
    return values

def parse_and_validate(cfg: Dict[str, Any], raw: Dict[str, Any]) -> Tuple[List[float], List[str]]:
    parsed: List[float] = []
    errors: List[str] = []
    for feat in cfg["features"]:
        label = feat["label"]
        ftype = feat["type"]
        val = raw[label]
        if ftype == "select":
            # already numeric (choices are ints)
            parsed.append(float(val))
            continue
        # text_input path
        if val.strip() == "":
            errors.append(f"{label}: missing")
            continue
        try:
            if ftype == "int":
                parsed.append(int(val))
            else:
                parsed.append(float(val))
        except ValueError:
            errors.append(f"{label}: invalid number '{val}'")
    return parsed, errors

def render_extra_help(text: str):
    if text:
        with st.expander("Feature Help / Definitions"):
            st.markdown(text)

def disease_page(name: str):
    cfg = DISEASES[name]
    st.header(name)
    render_extra_help(cfg.get("extra_help", ""))
    with st.form(f"{name}_form"):
        raw_values = render_feature_inputs(cfg)
        submitted = st.form_submit_button("Predict")
    if submitted:
        features, errs = parse_and_validate(cfg, raw_values)
        if errs:
            st.error("Validation errors:\n" + "\n".join(f"- {e}" for e in errs))
            return
        try:
            model = load_model(cfg["model_file"])
            pred, prob = get_prediction(model, features)
            label = cfg["positive_label"] if int(pred) == 1 else cfg["negative_label"]
            if prob is not None and len(prob) == 2:
                st.success(f"Prediction: {label} (probability={prob[1]:.2%})")
            else:
                st.success(f"Prediction: {label}")
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error during prediction: {e}")

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    with st.sidebar:
        selected = option_menu(
            APP_TITLE,
            list(DISEASES.keys()),
            icons=["activity", "heart", "person-fill", "clipboard2-pulse-fill"],
            default_index=0,
        )
        st.markdown("---")
        st.caption(DISCLAIMER)
    disease_page(selected)

if __name__ == "__main__":
    main()