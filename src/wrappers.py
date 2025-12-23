import torch
import numpy as np
from transformers import pipeline
from .config import RE_ID2LABEL, SPECIAL_TOKENS, DEVICE

# --- HÀM PHỤ TRỢ: GỘP BIO TAGS THÀNH ENTITY ---
def aggregate_entities(tokens, tags):
    """
    Input: 
      tokens = ['Tai', 'nạn', 'tại', 'Hà', 'Nội']
      tags   = ['O',   'O',   'O',   'B-LOC', 'I-LOC']
    Output:
      [{'word': 'Hà Nội', 'entity_group': 'LOC'}]
    """
    entities = []
    current_entity = None
    
    for token, tag in zip(tokens, tags):
        if tag == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
            continue
            
        # Tách B-LOC thành prefix=B, label=LOC
        parts = tag.split('-')
        label = parts[1] if len(parts) > 1 else tag
        prefix = parts[0] if len(parts) > 1 else ''
        
        if prefix == 'B':
            if current_entity:
                entities.append(current_entity)
            current_entity = {"word": token, "entity_group": label}
        elif prefix == 'I':
            if current_entity and current_entity['entity_group'] == label:
                current_entity['word'] += " " + token
            else:
                # Trường hợp I- nằm lẻ loi (coi như bắt đầu mới)
                if current_entity:
                    entities.append(current_entity)
                current_entity = {"word": token, "entity_group": label}
        else:
            # Trường hợp nhãn không có B/I (ít gặp nhưng đề phòng)
            if current_entity:
                entities.append(current_entity)
            current_entity = {"word": token, "entity_group": tag}
             
    if current_entity:
        entities.append(current_entity)
    return entities

# --- CÁC CLASS WRAPPER ---
class BasePredictor:
    def __init__(self, model_type):
        self.model_type = model_type

class NERPredictor(BasePredictor):
    def __init__(self, model_type, model, tokenizer=None, feature_extractor=None, label_map=None):
        super().__init__(model_type)
        self.model = model
        
        if model_type == 'DL':
            self.pipe = pipeline("token-classification", model=model, tokenizer=tokenizer, 
                                 aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1)
        else:
            self.feature_extractor = feature_extractor
            self.label_map = label_map 

    def predict(self, text):
        if self.model_type == 'DL':
            return self.pipe(text)
        
        else:
            # --- LOGIC CHO ML ---
            tokens = text.split()
            if not tokens: return []

            # === TRƯỜNG HỢP RIÊNG CHO CRF ===
            # Kiểm tra nếu là model CRF (sklearn_crfsuite.CRF)
            if "CRF" in str(type(self.model)) or hasattr(self.model, "tagger_"):
                # 1. Lấy features chuẩn format notebook (List of Dicts)
                features = self.feature_extractor.extract_crf_features(text)
                
                # 2. Predict
                # CRF predict nhận vào list các câu: [[feat1, feat2], [feat1, feat2]]
                # Nên ta phải bọc features vào 1 list: [features]
                if not features:
                    return []
                    
                try:
                    # Trả về list of lists of labels: [['B-LOC', 'O', ...]]
                    preds_list = self.model.predict([features])
                    
                    if len(preds_list) > 0:
                        preds = preds_list[0]
                    else:
                        preds = ["O"] * len(tokens)
                except Exception as e:
                    print(f"[ERROR CRF Predict]: {e}")
                    return []

            # === TRƯỜNG HỢP CHO LOGREG / SVM ===
            else:
                vectors = self.feature_extractor.vectorize_token_level(text)
                if len(vectors) == 0:
                    preds = []
                else:
                    pred_ids = self.model.predict(vectors)
                    preds = []
                    for pid in pred_ids:
                        label = self.label_map.get(pid) or self.label_map.get(str(pid)) or 'O'
                        preds.append(label)

            # Gộp kết quả
            min_len = min(len(tokens), len(preds))
            entities = aggregate_entities(tokens[:min_len], preds[:min_len])
            
            return entities

class REPredictor(BasePredictor):
    def __init__(self, model_type, model, tokenizer=None, feature_extractor=None, label_encoder=None):
        super().__init__(model_type)
        self.device = DEVICE
        self.model = model
        
        if model_type == 'DL':
            self.tokenizer = tokenizer
            # Thêm Special Tokens cho model DL (nếu chưa có)
            if self.tokenizer:
                num_added = self.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
                if num_added > 0:
                    self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.to(self.device)
            self.model.eval()
        else:
            # ML Model cần feature extractor để vector hóa text (có chứa tags)
            self.feature_extractor = feature_extractor
            self.label_encoder = label_encoder

    def predict(self, text):
        """
        Input: Text đã chèn thẻ Typed Markers. VD: "Tại <S:LOC> Hà Nội </S:LOC>..."
        """
        if self.model_type == 'DL':
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            pred_id = logits.argmax().item()
            # Map ID -> Label từ Config
            return RE_ID2LABEL.get(pred_id, "NO_RELATION")
        
        else:
            # ML: Vector hóa text (đã có tags) -> Predict
            if not self.feature_extractor:
                return "ERROR: Missing Feature Extractor"
                
            vec = self.feature_extractor.vectorize_sentence_level(text)
            pred_id = self.model.predict([vec])[0]
            
            # Map ID -> Label
            # Trường hợp dùng LabelEncoder của Sklearn
            if self.label_encoder and hasattr(self.label_encoder, 'inverse_transform'):
                return self.label_encoder.inverse_transform([pred_id])[0]
            
            # Trường hợp model trả về thẳng ID khớp với RE_ID2LABEL
            return RE_ID2LABEL.get(int(pred_id), "NO_RELATION")