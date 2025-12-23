import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from .config import MODEL_PATHS, DEVICE, SPECIAL_TOKENS

class PhoBERTFeatureExtractor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print(f"--- [INFO] Loading Vectorizer Base ({MODEL_PATHS['VECTORIZER_BASE']})...")
            cls._instance = super(PhoBERTFeatureExtractor, cls).__new__(cls)
            
            # Giữ use_fast=False để tương thích tốt với PhoBERT
            cls._instance.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATHS["VECTORIZER_BASE"], 
                use_fast=False 
            )
            
            if SPECIAL_TOKENS:
                cls._instance.tokenizer.add_special_tokens({'additional_special_tokens': SPECIAL_TOKENS})
            
            cls._instance.model = AutoModel.from_pretrained(MODEL_PATHS["VECTORIZER_BASE"])
            cls._instance.model.resize_token_embeddings(len(cls._instance.tokenizer))
            cls._instance.model.to(DEVICE)
            cls._instance.model.eval()
            
        return cls._instance

    def vectorize_token_level(self, text):
        """
        Dùng cho ML Models (CRF, LogReg, SVM).
        Logic: Manual Alignment (Tương thích use_fast=False)
        """
        tokens = text.split()
        if not tokens: return np.array([])
        
        # 1. Tokenize input
        inputs = self.tokenizer(
            tokens, 
            is_split_into_words=True, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.last_hidden_state[0].cpu().numpy() # [seq_len, 768]
        
        # 2. TẠO MAPPING THỦ CÔNG (Thay vì dùng word_ids())
        # Logic: Encode từng từ lẻ để biết nó tách thành bao nhiêu subwords
        wids = [None] # [CLS] luôn là None
        for i, token in enumerate(tokens):
            # add_special_tokens=False để chỉ lấy subwords của từ đó
            subwords = self.tokenizer.encode(token, add_special_tokens=False)
            wids.extend([i] * len(subwords))
            
        # Cắt hoặc thêm None cho khớp với độ dài thực tế của sequence (do truncation/padding/SEP)
        seq_len = inputs['input_ids'].shape[1]
        if len(wids) < seq_len:
            wids.extend([None] * (seq_len - len(wids))) # Fill [SEP] và padding
        else:
            wids = wids[:seq_len] # Cắt bớt nếu bị truncate
            
        # 3. Lấy embedding của subword đầu tiên cho mỗi word
        final_vectors = []
        seen_word_idx = set()
        
        for idx, word_id in enumerate(wids):
            # idx: vị trí trong chuỗi subwords (tương ứng với embeddings)
            # word_id: index của từ gốc (0, 1, 2...)
            if word_id is not None and word_id not in seen_word_idx:
                final_vectors.append(embeddings[idx])
                seen_word_idx.add(word_id)
        
        # 4. Xử lý trường hợp bị cắt cụt (Truncation)
        # Nếu câu quá dài, PhoBERT cắt bớt -> thiếu vector cho các từ cuối
        # Ta fill bằng vector 0 để tránh lỗi shape
        if len(final_vectors) < len(tokens):
            diff = len(tokens) - len(final_vectors)
            for _ in range(diff):
                final_vectors.append(np.zeros(768))
                
        return np.array(final_vectors)

    def vectorize_sentence_level(self, text):
        """Dùng cho RE (Mean Pooling)"""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean Pooling logic
        mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_emb = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        return (sum_emb / sum_mask).cpu().numpy().flatten()
    
    def extract_crf_features(self, text):
        """Tạo đặc trưng cho CRF (List of Dicts)"""
        vectors = self.vectorize_token_level(text)
        
        if len(vectors) == 0:
            return []

        sent_feats = []
        for vec in vectors:
            # Tạo dictionary đặc trưng: {'d0': 0.1, 'd1': -0.5...}
            feat_dict = {f'd{i}': v for i, v in enumerate(vec)}
            sent_feats.append(feat_dict)
            
        return sent_feats