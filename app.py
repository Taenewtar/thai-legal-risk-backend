import os
import re
import csv
from pathlib import Path
from datetime import datetime

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pythainlp.tokenize import word_tokenize

# --- CONFIG & CONSTANTS ---
LABELS = {
    0: "ไม่พบความเสี่ยงหมิ่นประมาททางอาญาชัดเจน",
    1: "เสี่ยงหมิ่นประมาทซึ่งหน้า (มาตรา 393)",
    2: "เสี่ยงหมิ่นประมาทธรรมดา (มาตรา 326)",
    3: "เสี่ยงหมิ่นประมาทโดยการโฆษณา (มาตรา 328)",
}

INSULT_KEYWORDS = [
    "เหี้ย", "ควย", "ส้นตีน", "สถุล", "ขยะสังคม", "เศษเดน", "เฮงซวย", "ตอแหล", 
    "ชาติชั่ว", "ระยำ", "อัปสรี้", "ชั่วช้า", "เลวทราม", "ต่ำช้า", "หน้าด้าน", 
    "โง่", "ปัญญาอ่อน", "ควาย", "สารเลว", "สวะ", "เดนสังคม", "บัตรซบ", "ถ่อย", "กาก", "ชัญไร"
]

RISKY_PREFIXES = {"แอบ", "กำลัง", "ชอบ", "เคย", "คิดจะ", "พยายาม"}

# --- MODEL LOADING ---
@st.cache_resource(show_spinner="กำลังโหลดโมเดลภาษาไทยจาก Hugging Face Hub...")
def load_huggingface_model():
    # กำหนด ID โมเดลเป๊ะๆ ป้องกันปัญหาเรื่องช่องว่างตัวอักษร
    base_model_name = "airesearch/wangchanberta-base-att-spm-thaigpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(LABELS),
        id2label={index: label for index, label in LABELS.items()},
        label2id={label: index for index, label in LABELS.items()},
        ignore_mismatched_sizes=True,
    )
    model.eval()
    return tokenizer, model

tokenizer, model = load_huggingface_model()

# --- STREAMLIT UI ---
st.set_page_config(page_title="ระบบวิเคราะห์ความเสี่ยงทางกฎหมายอาญา", layout="wide")
st.title("⚖️ ระบบวิเคราะห์ความเสี่ยงทางกฎหมายอาญา (Criminal Legal Risk Engine)")
st.caption("พัฒนาด้วยโมเดล WangchanBERTa สำหรับประเมินความเสี่ยงมาตรา 393, 326, 328, 327, 329")

text_input = st.text_area("ป้อนข้อความภาษาไทยที่ต้องการตรวจสอบความเสี่ยง:", height=150, placeholder="พิมพ์ข้อความที่นี่...")

if st.button("เริ่มต้นวิเคราะห์ข้อความ", type="primary"):
    if not text_input.strip():
        st.warning("กรุณาป้อนข้อความก่อนทำการวิเคราะห์")
    else:
        with st.spinner("กำลังวิเคราะห์ความเสี่ยง..."):
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0][pred_class].item() * 100

            st.success("วิเคราะห์สำเร็จ!")
            st.subheader(f"ผลการประเมิน: {LABELS.get(pred_class, 'ไม่ทราบผล')}")
            st.info(f"ระดับความมั่นใจของโมเดล: {confidence:.2f}%")
