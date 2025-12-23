import os
import joblib
import json  
from transformers import AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer
from .config import MODEL_PATHS, DEVICE
from .features import PhoBERTFeatureExtractor
from .wrappers import NERPredictor, REPredictor

class SystemLoader:
    def __init__(self):
        self.feature_extractor = None
        self.cached_models = {} 

    def _get_extractor(self):
        if not self.feature_extractor:
            self.feature_extractor = PhoBERTFeatureExtractor()
        return self.feature_extractor

    def load_ner_model(self, model_name):
        cache_key = f"NER_{model_name}"
        if cache_key in self.cached_models:
            return self.cached_models[cache_key]

        print(f"--- [LOAD] Đang tải NER: {model_name}...")
        path = MODEL_PATHS["NER"].get(model_name)
        
        if model_name == 'PHOBERT':
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForTokenClassification.from_pretrained(path).to(DEVICE)
            predictor = NERPredictor('DL', model, tokenizer=tokenizer)
        
        else:
            model = joblib.load(path)
            
            map_path = MODEL_PATHS["NER"]["LABEL_MAP"]
            
            with open(map_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "id2label" in data:
                raw_map = data["id2label"] # Dạng {"0": "O", "1": "B-LOC"...}
                label_map = {int(k): v for k, v in raw_map.items()}
            else:
                # Fallback nếu file cấu trúc khác
                label_map = data

            predictor = NERPredictor('ML', model, 
                                     feature_extractor=self._get_extractor(), 
                                     label_map=label_map)
        
        self.cached_models[cache_key] = predictor
        return predictor

    def load_re_model(self, model_name):
        cache_key = f"RE_{model_name}"
        if cache_key in self.cached_models:
            return self.cached_models[cache_key]

        print(f"--- [LOAD] Đang tải RE: {model_name}...")
        path = MODEL_PATHS["RE"].get(model_name)

        if model_name == 'PHOBERT':
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
            predictor = REPredictor('DL', model, tokenizer=tokenizer)
        else:
            model = joblib.load(path)
            meta_path = MODEL_PATHS["RE"]["METADATA"]
            metadata = joblib.load(meta_path)
            encoder = metadata.get('label_encoder') if isinstance(metadata, dict) else metadata
            
            predictor = REPredictor('ML', model, 
                                    feature_extractor=self._get_extractor(), 
                                    label_encoder=encoder)

        self.cached_models[cache_key] = predictor
        return predictor