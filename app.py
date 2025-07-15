import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import random


model_name = "JuniorThanh/t5_CoEdit_Stable"

max_input_length = 256

st.set_page_config(page_title="NLPerfect", layout="wide")
st_model_load = st.text('Đang nạp dữ liệu')


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gec = pipeline("text2text-generation", model=model_name, tokenizer=tokenizer)       # ép chạy trên CPU)
    return tokenizer, gec

tokenizer, gec = load_model()
st.success('Khởi tạo chương trình thành công!')
st_model_load.text("")

# Input section
if "corrected_text" not in st.session_state:
    st.session_state.corrected_text = ""


# Header
st.markdown("<h2 style='text-align:left'>Trường Đại học Mở TP.Hồ Chí Minh</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:left'>Giới thiệu chung</h3>", unsafe_allow_html=True)
st.write("""
Công cụ hỗ trợ kiểm tra ngữ pháp trong văn bản Tiếng Anh do nhóm NLPerfect thiết kế dựa trên huấn luyện từ hai mô hình CoEdit và T5, có khả năng phân tích những lỗi sai ngữ pháp từ cơ bản đến nâng cao. Đặc biệt, người dùng có thể sử dụng công cụ mà không lo bị giới hạn số lần dùng.
""")

st.markdown("<h3 style='text-align:left'>Hướng dẫn sử dụng</h3>", unsafe_allow_html=True)
html_instructions = """
<ol style="text-align:left; margin-left:1em;">
  <li>Nhập văn bản tiếng Anh trong cột "Văn Bản Đầu Vào" (Lưu ý: không nhập chữ tiếng Việt có dấu)</li>
  <li>Bấm kiểm tra ngữ pháp sau khi hoàn tất nhập liệu.</li>
  <li>Xem kết quả ở cột kết quả.</li>
</ol>
"""
st.markdown(html_instructions, unsafe_allow_html=True)

# Hai cột cho input và output
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### Văn Bản Đầu Vào")
    text_input = st.text_area("Hãy nhập văn bản vào khung bên dưới.", max_chars=256, help="Tối đa 256 ký tự.")


def generate_correction(input_text):
        # Tokenize input
    inputs = ["Fix grammar: " + input_text]

    outputs = gec(inputs,max_length=256)
    
    decoded_outputs = [out['generated_text'] for out in outputs]
    st.session_state.corrected_text = decoded_outputs[0].strip()


with col2:
    st.markdown("### Kết Quả")
    if st.session_state.get("result"):
        st.markdown(st.session_state["result"])
    else:
        st.markdown("_Kết quả kiểm tra sẽ hiển thị ở đây._")

# Nút "Kiểm tra ngữ pháp"
if st.button("Kiểm tra ngữ pháp"):
    if text_input.strip():  # tránh input rỗng
        generate_correction(text_input)
    with col2:
        if st.session_state.corrected_text:
            st.markdown(f"**{st.session_state.corrected_text}**")
        
st.markdown("<h3 style='text-align:left;margin-top:5%'>Khảo sát sau khi sử dụng</h3>", unsafe_allow_html=True)
st.write("""
Cảm ơn mọi người đã dành thời gian sử dụng công cụ của nhóm mình. Nếu mọi người có điều cần góp ý thì bấm vào nút bên dưới để thực hiện khảo sát nhé, nhóm mình xin chân thành cảm ơn.
""")
st.link_button("Khảo sát và phản hồi", "#", type="primary")

st.markdown("---")
st.markdown("<div style='text-align:center;font-weight:700'>Công cụ hỗ trợ kiểm tra ngữ pháp Tiếng Anh của nhóm NLPerfect.</div>", unsafe_allow_html=True)
