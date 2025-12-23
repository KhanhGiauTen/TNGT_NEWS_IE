import re
from pyvi import ViTokenizer
from .preprocessing import sliding_window_extract, clean_text_basic
from .config import VALID_RE_PAIRS

class TNGTPipeline:
    def __init__(self, ner_model, re_model):
        self.ner_predictor = ner_model
        self.re_predictor = re_model
        self.valid_pairs = VALID_RE_PAIRS

    def _prepare_input_typed(self, text, source_entity, target_entity):
        """Chèn thẻ <S:TYPE>... vào văn bản"""
        s_text = source_entity['word']
        s_label = source_entity['entity_group']
        s_start = source_entity.get('start')
        s_end = source_entity.get('end')

        o_text = target_entity['word']
        o_label = target_entity['entity_group']
        o_start = target_entity.get('start')
        o_end = target_entity.get('end')

        # Fallback tìm vị trí nếu thiếu
        if s_start is None: s_start = text.find(s_text)
        if s_end is None: s_end = s_start + len(s_text)
        if o_start is None: o_start = text.find(o_text)
        if o_end is None: o_end = o_start + len(o_text)

        if s_start == -1 or o_start == -1: return None

        tag_s_open, tag_s_close = f" <S:{s_label}> ", f" </S:{s_label}> "
        tag_o_open, tag_o_close = f" <O:{o_label}> ", f" </O:{o_label}> "

        spans = [(s_start, tag_s_open), (s_end, tag_s_close),
                 (o_start, tag_o_open), (o_end, tag_o_close)]
        spans.sort(key=lambda x: x[0], reverse=True)

        processed_text = text
        for idx, tag_str in spans:
            processed_text = processed_text[:idx] + tag_str + processed_text[idx:]
            
        return ViTokenizer.tokenize(processed_text)

    def _generate_pairs(self, entities):
        candidates = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if i == j: continue
                l1 = e1.get('entity_group', '').replace("B-", "").replace("I-", "")
                l2 = e2.get('entity_group', '').replace("B-", "").replace("I-", "")
                
                if (l1, l2) in self.valid_pairs:
                    candidates.append({"source": e1, "target": e2})
        return candidates

    def run(self, raw_text):
        cleaned_text = clean_text_basic(raw_text)
        windows = sliding_window_extract(cleaned_text, window_size=3, step_size=2)
        print(f"-> Đã chia văn bản thành {len(windows)} cửa sổ xử lý.")

        all_entities = []
        all_relations = []

        for idx, chunk in enumerate(windows):
            # NER
            entities = self.ner_predictor.predict(chunk)
            
            # Chuẩn hóa output NER và thêm vào list tổng
            for e in entities:
                if 'word' in e:
                    e['word'] = e['word'].replace("@@", "")

                lbl = (e.get('entity_group') or e.get('labels') or e.get('entity', '')).replace("B-", "").replace("I-", "")
                e['entity_group'] = lbl # Cập nhật lại để dùng cho RE
                all_entities.append({"text": e.get('word'), "label": lbl, "window_id": idx})

            # RE (Typed Markers)
            pairs = self._generate_pairs(entities)
            for p in pairs:
                re_input = self._prepare_input_typed(chunk, p['source'], p['target'])
                if re_input:
                    label = self.re_predictor.predict(re_input)
                    if label != 'NO_RELATION':
                        all_relations.append({
                            "source": p['source'].get('word'),
                            "target": p['target'].get('word'),
                            "relation": label,
                            "window_id": idx
                        })

        return self._post_processing(all_entities, all_relations)

    def _post_processing(self, entities, relations):
        # Logic thống kê đơn giản
        unique_entities = {}
        for e in entities:
            key = (e['text'], e['label'])
            unique_entities[key] = unique_entities.get(key, 0) + 1
        
        final_entities = [{"text": k[0], "label": k[1], "count": v} for k, v in unique_entities.items()]

        unique_rels = set()
        final_rels = []
        for r in relations:
            key = (r['source'], r['relation'], r['target'])
            if key not in unique_rels:
                unique_rels.add(key)
                final_rels.append({"subject": r['source'], "relation": r['relation'], "object": r['target']})
                
        return {"entities": final_entities, "relations": final_rels}