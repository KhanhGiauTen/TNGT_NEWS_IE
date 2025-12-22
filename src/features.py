import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .config import MODEL_PATHS, DEVICE

class PhoBERTFeatureExtractor:
    _instance = None

    def __new__(cls):
        # Singleton Pattern: Chỉ load PhoBERT Base 1 lần duy nhất
        if cls._instance is None:
            print(f"--- [INFO] Loading Vectorizer Base ({MODEL_PATHS['VECTORIZER_BASE']})...")
            cls._instance = super(PhoBERTFeatureExtractor, cls).__new__(cls)
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["VECTORIZER_BASE"])
            cls._instance.model = AutoModel.from_pretrained(MODEL_PATHS["VECTORIZER_BASE"]).to(DEVICE)
            cls._instance.model.eval()
        return cls._instance

    def vectorize_token_level(self, text):
        """Dùng cho NER ML: Trả về vector cho từng từ"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state[0].cpu().numpy()
        
        # Mapping đơn giản (Lấy token đầu của word)
        word_ids = inputs.word_ids()
        word_vectors = []
        seen_ids = set()
        for idx, word_id in enumerate(word_ids):
            if word_id is not None and word_id not in seen_ids:
                word_vectors.append(embeddings[idx])
                seen_ids.add(word_id)
        return np.array(word_vectors)

    def vectorize_sentence_level(self, text):
        """Dùng cho RE ML: Trả về 1 vector đại diện câu (CLS token)"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    
    def extract_crf_features(self, text):
        """Dùng cho CRF: Chuyển vector thành feature dict"""
        vectors = self.vectorize_token_level(text)
        features = []
        for vec in vectors:
            features.append({f'dim_{i}': v for i, v in enumerate(vec)})
        return [features]