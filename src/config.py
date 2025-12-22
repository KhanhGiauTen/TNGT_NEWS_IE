import os
import torch


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATHS = {
    # Dùng chung cho các model ML (SVM, LogReg...) để đảm bảo đồng nhất input
    "VECTORIZER_BASE": "vinai/phobert-base",
    
    "NER": {
        "PHOBERT": "Sura3607/tngt-ner-phobert",  
        
        # File phụ trợ
        "LABEL_MAP": os.path.join(MODEL_DIR, "ner/label_map.pkl"),
        
        # Models ML (.pkl)
        "LOGREG":    os.path.join(MODEL_DIR, "ner/logistic_regression.pkl"),
        "SVM":       os.path.join(MODEL_DIR, "ner/svm_model.pkl"),
        "CRF":       os.path.join(MODEL_DIR, "ner/crf_model.pkl"),
    },
    
    "RE": {
        "PHOBERT": "Sura3607/tngt-re-phobert",  
        
        # File phụ trợ
        "METADATA":  os.path.join(MODEL_DIR, "re/metadata.pkl"),
        
        # Models ML (.joblib)
        "LOGREG":    os.path.join(MODEL_DIR, "re/logistic_regression.joblib"),
        "SVM":       os.path.join(MODEL_DIR, "re/svm_model.joblib"),
        "RF":        os.path.join(MODEL_DIR, "re/random_forest.joblib"),
    }
}