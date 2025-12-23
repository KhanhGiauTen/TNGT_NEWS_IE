import pandas as pd
import re
import json
import os
import unicodedata

# Danh sách từ viết tắt cần bảo vệ dấu chấm
ABBREVIATIONS = {
    "TP.": "TP<PRD>", 
    "Tp.": "Tp<PRD>",
    "Mr.": "Mr<PRD>", 
    "Mrs.": "Mrs<PRD>",
    "Dr.": "Dr<PRD>", 
    "Th.S": "Th.S<PRD>", 
    "TS.": "TS<PRD>",
    "Q.": "Q<PRD>" # Quận
}

def clean_text_basic(text):
    """
    Làm sạch văn bản cơ bản & xử lý từ viết tắt.
    Bổ sung: Chuẩn hóa Unicode NFC để tránh lỗi tách từ của PhoBERT.
    """
    if pd.isna(text) or text == "":
        return ""
    
    text = str(text).strip()
    
    # --- THÊM DÒNG NÀY ---
    # Chuyển về dạng dựng sẵn (NFC) để đồng bộ với từ điển của PhoBERT
    text = unicodedata.normalize('NFC', text) 
    # ---------------------

    # Thay thế từ viết tắt
    for abbr, replacement in ABBREVIATIONS.items():
        text = text.replace(abbr, replacement)
        
    text = re.sub(r'\s+', ' ', text) # Gộp khoảng trắng
    text = re.sub(r'\[.*?\]', '', text) # Xóa text trong ngoặc vuông
    
    return text.strip()

def restore_abbreviations(text):
    """
    (Tùy chọn) Khôi phục lại dấu chấm cho từ viết tắt để người dùng đọc dễ hơn
    """
    if not text: return ""
    return text.replace("<PRD>", ".")

def split_sentences(text):
    """
    Tách câu dựa trên regex.
    Logic: Dấu kết thúc câu (.?!) + Khoảng trắng + Chữ in hoa
    """
    if not text:
        return []
    
    # Regex lookbehind: Tìm dấu chấm/hỏi/than, theo sau là khoảng trắng và chữ hoa
    # Thay thế bằng xuống dòng để split
    text = re.sub(r'(?<=[.?!])\s+(?=[A-Z])', '\n', text)
    
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    return sentences

def sliding_window_extract(text, window_size=3, step_size=2):
    """
    Cắt văn bản thành các cửa sổ trượt (Sliding Window).
    Input: Văn bản dài (String).
    Output: List các đoạn văn bản ngắn (List of Strings).
    """
    sentences = split_sentences(text)
    
    if len(sentences) <= window_size:
        return [restore_abbreviations(text)]

    windows = []
    num_sentences = len(sentences)
    
    for i in range(0, num_sentences, step_size):
        window_sents = sentences[i : i + window_size]
        
        # Ghép lại thành đoạn văn
        chunk = " ".join(window_sents)
        
        # Khôi phục dấu chấm (TP<PRD>HCM -> TP.HCM)
        chunk = restore_abbreviations(chunk)
        
        windows.append(chunk)
        
        if i + window_size >= num_sentences:
            break
            
    return windows


class DataPipeline:
    def __init__(self, input_path: str, output_path: str, save_table: bool = True, window_size: int = 3, step_size: int = 2):
        """
        input_path: File CSV gốc (chứa cột 'content').
        output_path: File JSON đầu ra cho Label Studio.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.save_table = save_table
        self.window_size = window_size
        self.step_size = step_size
        
        # Đường dẫn lưu file trung gian
        self.prep_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "data/preprocessed"
        os.makedirs(self.prep_dir, exist_ok=True)

    def create_import_label_studio(self):
        """
        Đọc CSV -> Clean -> Sliding Window -> Lưu JSON & CSV.
        """
        print(f"--- Bắt đầu xử lý dữ liệu từ: {self.input_path} ---")
        
        try:
            df = pd.read_csv(self.input_path)
            # Kiểm tra xem có cột content không, nếu không thử tìm cột khác
            if 'content' not in df.columns:
                # Fallback nếu tên cột khác (ví dụ 'NoiDung')
                possible_cols = [c for c in df.columns if 'content' in c.lower() or 'noidung' in c.lower()]
                if possible_cols:
                    df.rename(columns={possible_cols[0]: 'content'}, inplace=True)
                else:
                    raise ValueError("File CSV không có cột 'content'.")
        except Exception as e:
            print(f"Lỗi đọc file CSV: {e}")
            return

        print("Đang làm sạch văn bản (Cleaning)...")
        # Sử dụng hàm clean_text_basic 
        df['clean_content'] = df['content'].apply(clean_text_basic)
        
        cleaned_csv_path = os.path.join(
            self.prep_dir, 
            os.path.basename(self.input_path).replace(".csv", "_cleaned.csv")
        )
        df.to_csv(cleaned_csv_path, index=False, encoding='utf-8')
        print(f"-> Đã lưu file Cleaned CSV: {cleaned_csv_path}")

        print(f"Đang tạo Sliding Windows (Size={self.window_size}, Step={self.step_size})...")
        
        tasks = []          # List chứa dict cho JSON
        window_rows = []    # List chứa dict cho CSV split
        
        window_idx = 0      # ID định danh cho từng window

        for idx, row in df.iterrows():
            original_text = row['clean_content']
            article_id = row.get('id', idx) 
            
            chunks = sliding_window_extract(original_text, self.window_size, self.step_size)
            
            for chunk_text in chunks:
                chunk_id = f"{article_id}_{window_idx}"
                
                # Cấu trúc JSON cho Label Studio
                tasks.append({
                    "data": {
                        "text": chunk_text,
                        "ref_id": chunk_id,
                        "article_id": str(article_id)
                    }
                })

                # Cấu trúc CSV bảng (dễ nhìn)
                window_rows.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "article_id": article_id
                })
                window_idx += 1
        
        
        if self.save_table:
            split_csv_path = os.path.join(
                self.prep_dir,
                os.path.basename(self.input_path).replace(".csv", "_cleaned_split.csv")
            )
            pd.DataFrame(window_rows).to_csv(split_csv_path, index=False, encoding='utf-8')
            print(f"-> Đã lưu CLEANED_SPLIT CSV tại: {split_csv_path}")

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        print(f"\n✅ HOÀN TẤT XỬ LÝ.")
        print(f"- Tổng số bài báo gốc: {len(df)}")
        print(f"- Tổng số Windows (Tasks) tạo ra: {len(tasks)}")
        print(f"- File Output JSON: {self.output_path}")


# if __name__ == "__main__":
#     input_file = "data/raw/data_raw_400news.csv"  
#     output_file = "data/label_studio/import_tasks.json"
    
#     if os.path.exists(input_file):
#         pipeline = DataPipeline(input_file, output_file)
#         pipeline.create_import_label_studio()
#     else:
#         print(f"File {input_file} không tồn tại. Vui lòng kiểm tra đường dẫn.")