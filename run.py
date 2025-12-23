# run.py
import sys
import os

# Đảm bảo python nhìn thấy thư mục src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.loader import SystemLoader
from src.pipeline import TNGTPipeline

def bootstrap_system():
    """Hàm khởi tạo hệ thống: Load model vào RAM"""
    print("=== BOOTSTRAP: ĐANG KHỞI TẠO HỆ THỐNG ===")
    loader = SystemLoader()
    
    # 1. Load NER Model
    # Bạn có thể đổi thành 'PHOBERT' hoặc 'SVM' tại đây
    print("-> Loading NER Model...")
    ner_model = loader.load_ner_model("PHOBERT") 
    
    # 2. Load RE Model
    print("-> Loading RE Model...")
    re_model = loader.load_re_model("PHOBERT")
    
    # 3. Khởi tạo Pipeline với model đã load
    print("-> Initializing Pipeline...")
    pipeline = TNGTPipeline(ner_model, re_model)
    
    print("=== HỆ THỐNG SẴN SÀNG ===\n")
    return pipeline

def main():
    # Gọi hàm bootstrap
    pipeline = bootstrap_system()

    # Dữ liệu mẫu
    article = """
    Vào khoảng 15h30 chiều ngày 20/11, một vụ tai nạn giao thông nghiêm trọng đã xảy ra tại ngã tư Hàng Xanh, TP.HCM. 
    Xe tải mang BKS 29C-123.45 do tài xế Nguyễn Văn A điều khiển đã va chạm mạnh với xe khach. 
    Nạn nhân bị thương nặng được đưa đi cấp cứu.
    """

    # Chạy xử lý
    print("--- Input: Bài báo mẫu ---")
    result = pipeline.run(article)
    
    # In kết quả
    print("\n" + "="*40)
    print("KẾT QUẢ TRÍCH XUẤT")
    print("="*40)
    
    print(f"\n[ENTITIES] ({len(result['entities'])}):")
    for e in result['entities']:
        print(f" - [{e['label']}] {e['text']}")
        
    print(f"\n[RELATIONS] ({len(result['relations'])}):")
    for r in result['relations']:
        print(f" - {r['subject']} --{r['relation']}--> {r['object']}")

if __name__ == "__main__":
    main()