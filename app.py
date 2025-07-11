import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_name = "JuniorThanh/t5-base-medium-grammar-correction-ou"
max_input_length = 512

st.set_page_config(layout="centered")
st.header("Grammar Correction")

st_model_load = st.text('Loading model...')

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()
st.success('Model loaded!')
st_model_load.text("")

# Input section
if "corrected_text" not in st.session_state:
    st.session_state.corrected_text = ""

# Nhập liệu trực tiếp
text_input = st.text_area('Text to correct grammar for', height=300)

def generate_correction(input_text):
    # Tokenize input
    inputs = ["fix grammaticality in this sentence: " + input_text + "."]
    inputs = tokenizer(inputs, return_tensors="pt", truncation=True, max_length=max_input_length)

    outputs = model.generate(
    **inputs,
    num_beams=5,              # sử dụng beam search để tạo phản hồi chất lượng ổn định
    do_sample=False,          # không dùng sampling (loại bỏ ngẫu nhiên)
    repetition_penalty=2.0,   # tránh lặp từ
    no_repeat_ngram_size=3,   # không lặp cụm từ 3-gram
    length_penalty=1.0,       # độ dài phản hồi tương đương đầu vào
    max_length=inputs["input_ids"].size(1) + 5,  # chỉ dài hơn đầu vào tối đa 5 token
    early_stopping=True
)

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    st.session_state.corrected_text = decoded_outputs[0].strip()

# Button
if st.button('Fix grammar'):
    if text_input.strip():  # tránh input rỗng
        generate_correction(text_input)

# Display output
if st.session_state.corrected_text:
    st.subheader("Corrected Sentence")
    st.markdown(f"✅ **{st.session_state.corrected_text}**")
