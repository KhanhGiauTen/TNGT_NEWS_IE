import streamlit as st
import sys
import os
import pandas as pd
import graphviz
import time

# CẤU HÌNH ĐƯỜNG DẪN 
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.loader import SystemLoader
    from src.pipeline import TNGTPipeline
except ImportError as e:
    st.error(f"Lỗi import module: {e}")
    st.stop()

# CẤU HÌNH TRANG 
st.set_page_config(page_title="TNGT IE Demo", layout="wide", page_icon="⚡")

NER_MODELS_LIST = ["PHOBERT", "CRF", "SVM", "LOGREG"]
RE_MODELS_LIST = ["PHOBERT", "SVM", "RF", "LOGREG"]

# HÀM LOAD TOÀN BỘ MODEL (CACHED) 
@st.cache_resource(show_spinner=False)
def load_all_models_at_startup():
    """Load TẤT CẢ model vào RAM khi khởi động."""
    loader = SystemLoader()
    model_store = {
        "NER": {},
        "RE": {}
    }
    
    progress_bar = st.progress(0, text="Đang khởi tạo hệ thống...")
    total_steps = len(NER_MODELS_LIST) + len(RE_MODELS_LIST)
    step_count = 0

    #Load All NER Models
    for name in NER_MODELS_LIST:
        step_count += 1
        progress_bar.progress(step_count / total_steps, text=f"Đang tải NER Model: {name} ({step_count}/{total_steps})")
        try:
            model_store["NER"][name] = loader.load_ner_model(name)
        except Exception as e:
            print(f"Error loading NER {name}: {e}")
            model_store["NER"][name] = None

    #Load All RE Models
    for name in RE_MODELS_LIST:
        step_count += 1
        progress_bar.progress(step_count / total_steps, text=f"Đang tải RE Model: {name} ({step_count}/{total_steps})")
        try:
            model_store["RE"][name] = loader.load_re_model(name)
        except Exception as e:
            print(f"Error loading RE {name}: {e}")
            model_store["RE"][name] = None
            
    progress_bar.empty()
    return model_store

# GIAO DIỆN CHÍNH ---

with st.spinner('Đang tải toàn bộ dữ liệu vào RAM (Lần đầu sẽ mất khoảng 1-2 phút)...'):
    ALL_MODELS = load_all_models_at_startup()

st.sidebar.title("⚙️ Control Panel")

st.sidebar.subheader("Mô hình NER")
selected_ner_name = st.sidebar.selectbox("Chọn model NER:", NER_MODELS_LIST, index=0)

st.sidebar.subheader("Mô hình RE")
selected_re_name = st.sidebar.selectbox("Chọn model RE:", RE_MODELS_LIST, index=0)

ner_model = ALL_MODELS["NER"].get(selected_ner_name)
re_model = ALL_MODELS["RE"].get(selected_re_name)

# Khởi tạo Pipeline
if ner_model and re_model:
    pipeline = TNGTPipeline(ner_model, re_model)
else:
    st.error("Có lỗi khi load model. Vui lòng kiểm tra log.")
    st.stop()

# UI INPUT & OUTPUT ---
st.title("Hệ thống Trích xuất Thông tin TNGT")
st.caption("Demo load toàn bộ model tại thời điểm khởi động (Pre-load All)")

default_text = """Vào khoảng 15h30 chiều ngày 20/11, một vụ tai nạn giao thông nghiêm trọng đã xảy ra tại ngã tư Hàng Xanh, TP.HCM. 
Xe tải mang BKS 29C-123.45 do tài xế Nguyễn Văn A điều khiển đã va chạm mạnh với xe khách. 
Nạn nhân bị thương nặng được đưa đi cấp cứu."""

col_input, col_action = st.columns([3, 1])
with col_input:
    input_text = st.text_area("Nhập văn bản bài báo:", value=default_text, height=150)

with col_action:
    st.write("##")
    run_btn = st.button("Phân tích", type="primary", use_container_width=True)

if run_btn and input_text:
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Entities")
        ent_placeholder = st.empty()
        ent_placeholder.info("⏳ Đang chờ kết quả...")
        
    with col2:
        st.subheader("Relations")
        rel_placeholder = st.empty()
        rel_placeholder.info("⏳ Đang chờ kết quả...")
    
    st.divider()
    st.subheader("Visualization")
    vis_placeholder = st.empty()
    vis_placeholder.info("⏳ Đang chờ dữ liệu để vẽ biểu đồ...")

    # CHẠY PIPELINE
    try:
        with st.spinner('Đang chạy mô hình AI...'):
            start_time = time.time()
            result = pipeline.run(input_text)
            process_time = time.time() - start_time
            st.toast(f"Xử lý xong trong {process_time:.2f}s!", icon="🎉")

        
        # Cập nhật Entities
        with ent_placeholder.container():
            if result['entities']:
                df_ent = pd.DataFrame(result['entities'])
                st.dataframe(
                    df_ent[['text', 'label']].rename(columns={'text':'Text', 'label':'Loại'}), 
                    use_container_width=True, hide_index=True
                )
            else:
                st.warning("Không phát hiện thực thể.")
        
        # Cập nhật Relations
        with rel_placeholder.container():
            if result['relations']:
                df_rel = pd.DataFrame(result['relations'])
                st.dataframe(
                    df_rel.rename(columns={'subject':'Chủ thể', 'relation':'Quan hệ', 'object':'Đối tượng'}),
                    use_container_width=True, hide_index=True
                )
            else:
                st.warning("Không phát hiện quan hệ.")

        # Vẽ biểu đồ 
        with vis_placeholder.container():
            if result['entities'] or result['relations']:
                with st.status("Đang hiển thị biểu đồ tương tác...", expanded=True) as status:
                    graph = graphviz.Digraph()
                    graph.attr(rankdir='LR')
                    
                    colors = {
                        'LOC': '#ffebee', 'TIME': '#e8f5e9', 'VEH': '#e3f2fd',
                        'PER_DRIVER': '#fff3e0', 'PER_VICTIM': '#f3e5f5', 
                        'EVENT': '#eceff1', 'CAUSE': '#fbe9e7', 'CONSEQUENCE': '#fff8e1'
                    }
                    
                    added_nodes = set()
                    
                    for ent in result['entities']:
                        node_id = ent['text']
                        if node_id not in added_nodes:
                            lbl = f"{ent['text']}\n({ent['label']})"
                            c = colors.get(ent['label'], 'white')
                            graph.node(node_id, label=lbl, style='filled', fillcolor=c, shape='box', rx='5', ry='5')
                            added_nodes.add(node_id)
                    
                    for rel in result['relations']:
                        if rel['subject'] not in added_nodes:
                            graph.node(rel['subject'], label=rel['subject'])
                            added_nodes.add(rel['subject'])
                        if rel['object'] not in added_nodes:
                            graph.node(rel['object'], label=rel['object'])
                            added_nodes.add(rel['object'])
                        graph.edge(rel['subject'], rel['object'], label=rel['relation'], fontsize='10')

                    st.graphviz_chart(graph)
                    
                    status.update(label="Vẽ biểu đồ thành công!", state="complete", expanded=False)
            else:
                st.info("Chưa đủ dữ liệu để vẽ biểu đồ.")

    except Exception as e:
        st.error(f"Lỗi xử lý: {e}")
        st.exception(e)