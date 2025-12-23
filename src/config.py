import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RE_ID2LABEL = {
    0: "NO_RELATION",
    1: "CAUSED_BY",
    2: "HAPPENED_ON",
    3: "HAS_CONSEQUENCE",
    4: "INVOLVED",
    5: "LOCATED_AT"
}
RE_LABEL2ID = {v: k for k, v in RE_ID2LABEL.items()}

# Các cặp quan hệ hợp lệ 
VALID_RE_PAIRS = {
    ('EVENT', 'LOC'), 
    ('EVENT', 'TIME'), 
    ('EVENT', 'VEH'), ('PER_DRIVER', 'VEH'), ('PER_VICTIM', 'VEH'),
    ('EVENT', 'CAUSE'),
    ('EVENT', 'CONSEQUENCE')
}

# Special Tokens cho Typed Entity Markers
ENTITY_TYPES = ['TIME', 'CAUSE', 'PER_VICTIM', 'LOC', 'CONSEQUENCE', 'EVENT', 'ORG', 'VEH', 'PER_DRIVER']
SPECIAL_TOKENS = []
for lbl in ENTITY_TYPES:
    SPECIAL_TOKENS.extend([f"<S:{lbl}>", f"</S:{lbl}>", f"<O:{lbl}>", f"</O:{lbl}>"])


MODEL_PATHS = {
    "VECTORIZER_BASE": "vinai/phobert-base",
    
    "NER": {
        "PHOBERT": "Sura3607/tngt-ner-phobert",  
        "LABEL_MAP": os.path.join(MODEL_DIR, "ner/label_map.json"),
        "LOGREG":    os.path.join(MODEL_DIR, "ner/logistic_regression.pkl"),
        "SVM":       os.path.join(MODEL_DIR, "ner/svm_model.pkl"),
        "CRF":       os.path.join(MODEL_DIR, "ner/crf_model.pkl"),
    },
    
    "RE": {
        "PHOBERT": "Sura3607/tngt-re-phobert",  
        "METADATA":  os.path.join(MODEL_DIR, "re/metadata.pkl"),
        "LOGREG":    os.path.join(MODEL_DIR, "re/lr_model.joblib"),
        "SVM":       os.path.join(MODEL_DIR, "re/svm_model.joblib"),
        "RF":        os.path.join(MODEL_DIR, "re/rf_model.joblib"),
    }
}