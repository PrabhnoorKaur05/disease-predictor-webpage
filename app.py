# app_ml.py (Hybrid Stacking + Realistic Predictions + Recommendation + Explanation)

import os
import pickle
import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

CSV_PATH = "Disease and symptoms dataset.csv"
MODEL_PATH = "clf.pkl"
SYMPTOMS_PATH = "symptoms.pkl"

# 1) Load dataset
df = pd.read_csv(CSV_PATH)
if "diseases" not in df.columns:
    raise ValueError("CSV must contain a 'diseases' column.")

all_symptoms = [c for c in df.columns if c != "diseases"]

# 2) Train (first run) or load model
if not os.path.exists(MODEL_PATH) or not os.path.exists(SYMPTOMS_PATH):
    print("🔄 Training hybrid stacking model (first run)...")
    n_sample = min(10000, len(df))
    df_sample = df.sample(n_sample, random_state=42)

    X = df_sample.drop(columns=["diseases"])
    y = df_sample["diseases"]

    # Base models
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)),
        ('svm', SVC(probability=True, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ]

    # Meta model
    meta_model = LogisticRegression(max_iter=1000, random_state=42)

    # Stacking classifier
    clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        stack_method='predict_proba',
        n_jobs=-1
    )
    clf.fit(X, y)

    # Save model and symptom order
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(SYMPTOMS_PATH, "wb") as f:
        pickle.dump(list(X.columns), f)

    print("✅ Hybrid model trained & saved.")
else:
    print("✅ Loading saved hybrid model...")
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(SYMPTOMS_PATH, "rb") as f:
        all_symptoms = pickle.load(f)

# 3) Auto-detect symptom categories
def build_categories(symptom_cols):
    lower_map = {s: s.lower() for s in symptom_cols}
    patterns = {
        "Respiratory Issues": ["cough","breath","dyspnea","wheeze","lung","sputum","throat","chest pain","chest tight","sore throat"],
        "Neurological Issues": ["headache","head pain","migraine","dizz","vertigo","seiz","confus","numb","tingl","tremor","insom","memory"],
        "Skin Issues": ["skin","rash","itch","blister","red","dry","lesion","hive","acne","psoria","dermat"],
        "Digestive Issues": ["nausea","vomit","diarrh","stomach","abdom","constipat","gastric","ulcer","indigest","acid"],
        "Musculoskeletal Issues": ["back","joint","muscle","stiff","swelling","ache","spasm","neck","shoulder","knee","hip","ankle","wrist","elbow"],
        "Cardiac Issues": ["heart","palpit","card","pulse","angina","chest pain"],
        "ENT Issues": ["ear","nose","nasal","sinus","throat","sneeze","congest"],
        "Psychological Issues": ["depress","psychot","hallucin","mood","stress","anxiety","panic","phobia"],
        "General/Constitutional": ["fatigue","fever","sweat","chill","weight","weak","tired","malaise","appetite","night sweat"],
    }

    categories = {k: [] for k in patterns.keys()}
    for s in symptom_cols:
        ls = lower_map[s]
        matched = False
        for cat, keys in patterns.items():
            if any(k in ls for k in keys):
                categories[cat].append(s)
                matched = True
        if not matched:
            categories.setdefault("Other / Unclassified", []).append(s)
    for cat in categories:
        categories[cat] = sorted(categories[cat])
    return categories

SYMPTOM_CATEGORIES = build_categories(all_symptoms)

def symptoms_for_categories(selected_categories):
    if not selected_categories:
        return []
    s = set()
    for cat in selected_categories:
        s.update(SYMPTOM_CATEGORIES.get(cat, []))
    return sorted(s)

# 4) Gradio callbacks
def get_symptoms(selected_categories):
    return gr.update(choices=symptoms_for_categories(selected_categories), value=[])

def predict_disease(selected_categories, selected_symptoms):
    if not selected_categories:
        return "⚠️ Please select at least one category."
    if not selected_symptoms:
        return "⚠️ Please select at least one symptom."

    feature_index = {s: i for i, s in enumerate(all_symptoms)}
    x = [0] * len(all_symptoms)
    for s in selected_symptoms:
        if s in feature_index:
            x[feature_index[s]] = 1

    probs = clf.predict_proba([x])[0]
    labels = clf.classes_
    pairs = sorted(zip(labels, probs), key=lambda t: t[1], reverse=True)

    top_diseases = [pairs[0][0]]
    if len(pairs) > 1 and pairs[1][1] >= 0.5 * pairs[0][1]:
        top_diseases.append(pairs[1][0])

    out = ["🩺 Most Likely Disease(s):"]
    for d in top_diseases:
        out.append(f"- {d}")

    out.append("\n⚠️ Please consult a medical professional for confirmation.")
    return "\n".join(out)

# 5) Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🩺 Disease Prediction Model \n"
                "Pick one or more **categories**, then select relevant **symptoms**.")

    with gr.Row():
        category = gr.CheckboxGroup(
            choices=list(SYMPTOM_CATEGORIES.keys()),
            label="Select Symptom Categories"
        )
        symptom_box = gr.CheckboxGroup(choices=[], label="Select Symptoms (filtered by categories)")

    with gr.Row():
        predict_btn = gr.Button("🔍 Predict Disease")
    output = gr.Textbox(label="Prediction Result", lines=8)

    gr.Markdown("""
**How this model predicts diseases:**  

1. You select symptom categories and symptoms.  
2. The app converts your selection into a format the model understands.  
3. Three models (Random Forest, SVM, Logistic Regression) predict probabilities for each disease.  
4. A meta-model combines these predictions to give the final result.  
5. The top disease is always shown; a second disease appears only if it is likely.  
6. Predictions are suggestions; always consult a doctor for confirmation.
""")

    category.change(fn=get_symptoms, inputs=category, outputs=symptom_box)
    predict_btn.click(fn=predict_disease, inputs=[category, symptom_box], outputs=output)

# 6) Launch
if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=7861, debug=True)
