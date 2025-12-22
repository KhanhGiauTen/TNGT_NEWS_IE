import torch
import numpy as np
from transformers import pipeline

# --- LỚP CHA (BASE) ---
class BasePredictor:
    def __init__(self, model_type):
        self.model_type = model_type  # 'ML' hoặc 'DL'

# --- WRAPPER CHO NER ---
class NERPredictor(BasePredictor):
    def __init__(self, model_type, model, tokenizer=None, feature_extractor=None, label_map=None):
        super().__init__(model_type)
        self.model = model
        
        if model_type == 'DL':
            # Setup Pipeline cho Deep Learning
            self.pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, 
                                 aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
        else:
            # Setup cho Machine Learning
            self.feature_extractor = feature_extractor
            self.label_map = label_map # Dict {0: 'O', 1: 'B-LOC'}

    def predict(self, text):
        """Input: Text string -> Output: List of entities"""
        if self.model_type == 'DL':
            # Xử lý DL: Pipeline tự lo mọi thứ
            return self.pipe(text)
        
        else:
            # Xử lý ML: Tokenize -> Vectorize -> Predict -> Map Label
            # 1. Vectorize
            if hasattr(self.model, "predict_marginals") or "CRF" in str(type(self.model)):
                # Xử lý riêng cho CRF (nếu dùng sklearn-crfsuite)
                vectors = self.feature_extractor.extract_crf_features(text)
            else:
                vectors = self.feature_extractor.vectorize_token_level(text)
            
            # 2. Predict (trả về id hoặc label)
            preds = self.model.predict(vectors)
            
            # 3. Map ID sang Label (nếu model trả về số)
            # Giả sử preds là list số [0, 1, 0...]
            if self.label_map and isinstance(preds[0], (int, np.integer)):
                decoded_preds = [self.label_map.get(p, 'O') for p in preds]
            else:
                decoded_preds = preds # Model đã trả về string (ví dụ CRF)
                
            # 4. Format lại output cho giống DL (để App dễ hiển thị)
            # Phần này cần logic gộp BIO tags thành entity, ở đây làm đơn giản để demo
            return [{"word": text, "labels": decoded_preds}]

# --- WRAPPER CHO RE ---
class REPredictor(BasePredictor):
    def __init__(self, model_type, model, tokenizer=None, feature_extractor=None, label_encoder=None):
        super().__init__(model_type)
        self.model = model
        
        if model_type == 'DL':
            self.tokenizer = tokenizer
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.feature_extractor = feature_extractor
            self.label_encoder = label_encoder # LabelEncoder object

    def predict(self, text):
        """Input: Text string (hoặc cặp entity) -> Output: Relation Label"""
        if self.model_type == 'DL':
            # Xử lý DL
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            pred_id = logits.argmax().item()
            return self.model.config.id2label[pred_id]
        
        else:
            # Xử lý ML
            vec = self.feature_extractor.vectorize_sentence_level(text)
            # Sklearn cần input shape (1, n_features)
            pred_id = self.model.predict([vec])[0]
            
            # Decode từ số sang chữ (0 -> 'CAUSED_BY')
            if self.label_encoder:
                # Nếu là LabelEncoder object
                if hasattr(self.label_encoder, 'inverse_transform'):
                    return self.label_encoder.inverse_transform([pred_id])[0]
                # Nếu là dict
                elif isinstance(self.label_encoder, dict):
                    return self.label_encoder.get(pred_id, "UNKNOWN")
            
            return pred_id