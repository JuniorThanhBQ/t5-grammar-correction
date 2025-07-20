import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import random


model_name = "grammarly/coedit-large"

max_input_length = 512

st.set_page_config(page_title="NLPerfect", layout="wide")
st_model_load = st.text('Loading data')


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gec = pipeline("text2text-generation", model=model_name, tokenizer=tokenizer)
    spell_corrector = pipeline( "text2text-generation",model="oliverguhr/spelling-correction-english-base",tokenizer=tokenizer)
    return tokenizer, gec, spell_corrector

tokenizer, gec, spell_corrector = load_model()
st.success('Program initialized successfully!')
st_model_load.text("")

# Input section
if "corrected_text" not in st.session_state:
    st.session_state.corrected_text = ""


# Header
st.markdown("<h2 style='text-align:left'>Ho Chi Minh City Open University</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:left'>General Introduction</h3>", unsafe_allow_html=True)
st.write("""
This English grammar checking tool is developed by the NLPerfect team based on the CoEdit and T5 models, capable of analyzing grammar errors from basic to advanced levels. Notably, users can use this tool without worrying about usage limits.
""")

st.markdown("<h3 style='text-align:left'>User Guide</h3>", unsafe_allow_html=True)
html_instructions = """
<ol style="text-align:left; margin-left:1em;">
  <li>Enter English text in the "Input Text" column (Note: do not enter Vietnamese with diacritics)</li>
  <li>Click the grammar check button after finishing your input.</li>
  <li>View the result in the result column.</li>
</ol>
"""
st.markdown(html_instructions, unsafe_allow_html=True)

# Two columns for input and output
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("### Input Text")
    text_input = st.text_area("Please enter your text in the box below.", max_chars=1024, help="Maximum 1024 characters.", height=350)


def generate_correction(input_text):
        # Tokenize input
    inputs = ["Fix grammar: " + input_text]

    spell_checked_output = spell_corrector(inputs,max_length=256)
    outputs = gec(spell_checked_output,max_length=256)
    
    decoded_outputs = [out['generated_text'] for out in outputs]
    st.session_state.corrected_text = decoded_outputs[0].strip()


with col2:
    st.markdown("### Result")
    if st.session_state.get("result"):
        st.markdown(st.session_state["result"])
    else:
        st.markdown("_The check result will appear here._")

# "Check Grammar" button
if st.button("Check Grammar"):
    if text_input.strip():  # avoid empty input
        generate_correction(text_input)
    with col2:
        if st.session_state.corrected_text:
            st.markdown(f"**{st.session_state.corrected_text}**")
        
st.markdown("<h3 style='text-align:left;margin-top:5%'>Post-Use Survey</h3>", unsafe_allow_html=True)
st.write("""
Thank you for taking the time to use our tool. If you have any feedback, please click the button below to take the survey. We sincerely appreciate it.
""")
st.link_button("Survey and Feedback", "#", type="primary")

st.markdown("---")
st.markdown("<div style='text-align:center;font-weight:700'>English Grammar Checking Tool by the NLPerfect team.</div>", unsafe_allow_html=True)
