import csv
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from pythainlp.tag import pos_tag
from pythainlp.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_PATH = Path(__file__).resolve().parent
MODEL_ID = os.getenv("LEGAL_MODEL_ID", "airesearch/wangchanberta-base-att-spm-uncased")
LABELS = {0: "ทั่วไป/ติชม", 1: "ดูหมิ่น", 2: "หมิ่นประมาท ม.326"}

LEGAL_SECTIONS_DB = {
    "มาตรา 393": {"title": "ดูหมิ่น", "summary": "ถ้อยคำดูถูก เหยียดหยาม หรือทำให้อับอาย", "elements": ["มีการดูถูกหรือเหยียดหยาม", "ต้องพิจารณาบริบทและเจตนา"]},
    "มาตรา 326": {"title": "หมิ่นประมาททั่วไป", "summary": "ใส่ความผู้อื่นต่อบุคคลที่สามจนมีแนวโน้มทำให้เสียชื่อเสียง", "elements": ["มีผู้ถูกกล่าวถึง", "มีบุคคลที่สามรับรู้", "ข้อความอาจทำให้เสียชื่อเสียง ถูกดูหมิ่น หรือถูกเกลียดชัง"]},
    "มาตรา 328": {"title": "หมิ่นประมาทโดยการโฆษณา", "summary": "เผยแพร่ข้อความหรือสื่อให้บุคคลทั่วไปเข้าถึงได้", "elements": ["เผยแพร่ต่อสาธารณะ", "มีลักษณะใส่ความตามมาตรา 326"]},
    "มาตรา 327": {"title": "หมิ่นประมาทผู้ตาย", "summary": "กล่าวถึงผู้ตายจนกระทบชื่อเสียงของบุคคลในครอบครัวตามกฎหมาย", "elements": ["ผู้ถูกกล่าวถึงเป็นผู้ตาย", "กระทบชื่อเสียงหรือความรู้สึกของบุคคลที่กฎหมายคุ้มครอง"]},
    "มาตรา 329": {"title": "แสดงความคิดเห็นโดยสุจริต", "summary": "การติชมโดยสุจริตหรือปกป้องสิทธิโดยชอบอาจเป็นข้อพิจารณา", "elements": ["มีความสุจริต", "ใช้ถ้อยคำเท่าที่จำเป็น", "เกี่ยวข้องกับการปกป้องสิทธิหรือประโยชน์สาธารณะ"]},
}

# คลังคำเป็นสัญญาณประกอบเท่านั้น ไม่ใช่รายการตัดสินความผิด
DEFAMATION_KEYWORDS = [
    "ขายยา", "ขายยาบ้า", "ค้ายา", "แอบขายยา", "เอเย่นต์ยา", "ค้ายาเสพติด", "ฟอกเงิน", "อุ้มฆ่า", "บ่อน", "โต๊ะบอล", "เจ้ามือหวย", "ซูเอี๋ย",
    "โกง", "โกงเงิน", "ทุจริต", "ยักยอก", "แอบยักยอก", "รับสินบน", "คอรัปชั่น", "คอร์รัปชัน", "กินส่วนต่าง", "แดกงบ", "ต้มตุ๋น", "ตบทรัพย์", "รีดไถ", "ไถเงิน", "ยัดข้อหา", "ฉ้อโกง", "หลอกลวง", "ขโมย",
    "เป็นชู้", "แอบเป็นชู้", "ชู้", "เมียน้อย", "เมียเก็บ", "เต้าไต่", "ชู้สาว", "มั่ว", "ขายตัว", "มั่วผู้ชาย", "มั่วผู้หญิง", "นอกใจ", "มีชู้", "ค้าประเวณี", "ขายบริการ", "อีตัว", "กะหรี่", "แมงดา",
]
INSULT_KEYWORDS = [
    "เหี้ย", "ควย", "ส้นตีน", "สถุล", "ขยะสังคม", "เศษเดน", "เฮงซวย", "ตอแหล", "ชาติชั่ว", "ระยำ", "อัปสรี", "ชั่วช้า", "เลวทราม", "ต่ำช้า", "หน้าด้าน", "โง่", "ปัญญาอ่อน", "ควาย", "สารเลว", "สวะ", "เดนสังคม", "บัดซบ", "ถ่อย", "กาก", "จัญไร", "ไร้ค่า", "หน้าตัวเมีย",
]
RISKY_PREFIXES = {"แอบ", "กำลัง", "ชอบ", "เคย", "คิดจะ", "พยายาม"}
EXAMPLES = [
    "สั่งอาหารไปสองชั่วโมงแล้ว ยังไม่ได้รับอาหารเลย บริการช้ามาก",
    "สันดานโกงแบบนี้ อย่าไปทำธุรกิจด้วยเด็ดขาด",
    "หัวหน้าฝ่ายบัญชีบริษัทนี้ยักยอกเงินบริษัทไปใช้ส่วนตัว",
]
COLLECTED_PATH = PROJECT_PATH / "collected_dataset.csv"
AUGMENTED_PATH = PROJECT_PATH / "train_dataset_augmented.csv"
COLLECTED_FIELDS = ["timestamp", "text", "predicted_class", "confidence", "extracted_keywords"]
AUGMENTED_FIELDS = ["timestamp", "user_text", "predicted_sections", "correct_label"]


@st.cache_resource(show_spinner="กำลังดาวน์โหลดและโหลด WangchanBERTa จาก Hugging Face Hub...")
def load_huggingface_model():
    base_model_name = "pythainlp/wangchanberta-base-att-spm-thaigpt2"
    
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


def tokens_for(text):
    return word_tokenize(str(text), engine="newmm")


def tagged_for(text):
    tokens = tokens_for(text)
    try:
        return tokens, pos_tag(tokens, corpus="orchid", tagset="universal")
    except Exception:
        return tokens, pos_tag(tokens, corpus="orchid")


def pos_risk_terms(text):
    _, tagged = tagged_for(text)
    terms = []
    for token, tag in tagged:
        tag_text = str(tag).upper()
        if tag_text in {"VERB", "ADJ"} or tag_text.startswith(("V", "A", "JJ")):
            if len(token.strip()) > 1:
                terms.append(token)
    return list(dict.fromkeys(terms))


def risky_phrases(text):
    tokens, tagged = tagged_for(text)
    phrases = []
    for index, (_, tag) in enumerate(tagged):
        tag_text = str(tag).upper()
        if tag_text not in {"VERB", "ADJ"} and not tag_text.startswith(("V", "A", "JJ")):
            continue
        start = index - 1 if index and tokens[index - 1] in RISKY_PREFIXES else index
        phrase = "".join(tokens[start:min(len(tokens), index + 3)]).strip()
        if len(phrase) > 2:
            phrases.append(phrase)
    return list(dict.fromkeys(phrases))


def find_keywords(text):
    keywords = [word for word in sorted(set(DEFAMATION_KEYWORDS + INSULT_KEYWORDS), key=len, reverse=True) if word in text]
    if not keywords:
        keywords = risky_phrases(text) + pos_risk_terms(text)
    return list(dict.fromkeys(keywords))


def selected_sections(predicted, text):
    if predicted == 1:
        return ["มาตรา 393"]
    if predicted == 2:
        public_markers = ("โพสต์", "เฟซบุ๊ก", "facebook", "เพจ", "ออนไลน์", "อินเทอร์เน็ต", "สาธารณะ", "แชร์", "กลุ่ม")
        return ["มาตรา 328"] if any(marker in text.lower() for marker in public_markers) else ["มาตรา 326"]
    return ["มาตรา 329"]


def analyze_text(text):
    cleaned = str(text).strip()
    if not cleaned:
        raise ValueError("กรุณาพิมพ์ข้อความก่อนตรวจสอบ")
    if len(cleaned) < 3:
        raise ValueError("ข้อความสั้นเกินไป ระบบคัดกรองได้ไม่ดีพอ")
    if not any("\u0e00" <= char <= "\u0e7f" for char in cleaned):
        raise ValueError("กรุณาใช้ข้อความภาษาไทย")
    processed = " <_> ".join(tokens_for(cleaned))
    inputs = tokenizer(processed, return_tensors="pt", truncation=True, max_length=128)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()
    predicted = int(np.argmax(probabilities))
    return cleaned, predicted, float(probabilities[predicted]), probabilities, find_keywords(cleaned)


def write_collected(text, predicted, confidence, keywords):
    exists = COLLECTED_PATH.exists() and COLLECTED_PATH.stat().st_size > 0
    with COLLECTED_PATH.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLLECTED_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({"timestamp": datetime.now(timezone.utc).isoformat(), "text": text, "predicted_class": predicted, "confidence": f"{confidence:.6f}", "extracted_keywords": "|".join(keywords)})


def collection_count():
    if not COLLECTED_PATH.exists():
        return 0
    with COLLECTED_PATH.open(encoding="utf-8-sig", newline="") as file:
        return sum(1 for _ in csv.DictReader(file))


def append_feedback(text, predicted, correct_label):
    correct_id = next((index for index, label in LABELS.items() if label == correct_label), None)
    if correct_id is None:
        raise ValueError("ไม่พบ Class ที่เลือก")
    exists = AUGMENTED_PATH.exists() and AUGMENTED_PATH.stat().st_size > 0
    with AUGMENTED_PATH.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=AUGMENTED_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({"timestamp": datetime.now(timezone.utc).isoformat(), "user_text": text.strip(), "predicted_sections": ", ".join(selected_sections(int(predicted), text)), "correct_label": correct_id})


st.set_page_config(page_title="Criminal Legal Risk Engine", page_icon="⚖️", layout="wide")
st.title("⚖️ ระบบวิเคราะห์ความเสี่ยงทางกฎหมาย (Criminal Legal Risk Engine)")
st.caption(f"โมเดล: WangchanBERTa จาก Hugging Face Hub ({MODEL_ID}) | ผลคัดกรองเบื้องต้น ไม่ใช่คำวินิจฉัยทางกฎหมาย")

if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "example_text" not in st.session_state:
    st.session_state.example_text = ""

with st.sidebar:
    st.header("📝 ข้อความ")
    text_input = st.text_area("ข้อความที่ต้องการตรวจสอบ", value=st.session_state.example_text, height=220, placeholder="พิมพ์ข้อความภาษาไทย...")
    if st.button("🎲 สุ่มข้อความตัวอย่าง", use_container_width=True):
        st.session_state.example_text = str(np.random.choice(EXAMPLES))
        st.rerun()
    st.caption("ตัวอย่างข้อความ")
    for example in EXAMPLES:
        st.code(example, language=None)
    analyze_button = st.button("🔍 เริ่มวิเคราะห์ข้อความ", type="primary", use_container_width=True)

if analyze_button:
    try:
        analysis = analyze_text(text_input)
        write_collected(analysis[0], analysis[1], analysis[2], analysis[4])
        st.session_state.analysis = analysis
        st.success("วิเคราะห์และสะสมข้อมูลเรียบร้อย")
    except ValueError as error:
        st.error(str(error))

analysis = st.session_state.analysis
if analysis:
    cleaned, predicted, confidence, probabilities, keywords = analysis
    sections = selected_sections(predicted, cleaned)
    probability_text = "\n".join(f"- {LABELS.get(index, f'คลาส {index}')}: {float(value) * 100:.2f}%" for index, value in enumerate(probabilities))
    keyword_text = "\n".join(f"- `{keyword}`" for keyword in keywords) if keywords else "ไม่พบคำตรงจากคลัง จึงใช้ผล POS Tagging เป็นคำตั้งข้อสังเกต"
    tab1, tab2, tab3, tab4 = st.tabs(["📋 สรุปผล & กฎหมายที่เกี่ยวข้อง", "🔍 เหตุผล & วิเคราะห์เชิงลึก", "💡 แนวทางรับมือ", "📚 เกร็ดความรู้กฎหมาย"])
    with tab1:
        st.markdown(f"**ผลโมเดล:** {LABELS[predicted]}")
        st.markdown(f"**มาตราที่ควรตรวจต่อ:** {', '.join(sections)}")
        st.markdown(f"**ความมั่นใจ:** {confidence * 100:.2f}%")
        st.markdown("### ความน่าจะเป็นรายคลาส")
        st.markdown(probability_text)
    with tab2:
        st.markdown("### คำ/วลีที่ระบบตั้งข้อสังเกต")
        st.markdown(keyword_text)
        st.warning("คำหรือวลีเป็นเพียงสัญญาณประกอบ ต้องพิจารณาบุคคลที่สาม เจตนา และบริบททั้งหมด")
    with tab3:
        st.warning("ควรตรวจสอบข้อความฉบับเต็ม บริบท และหลักฐานกับผู้เชี่ยวชาญ")
        st.markdown("**สำหรับผู้เสียหาย**\n\n1. เก็บข้อความ URL ภาพหน้าจอ วันเวลา และพยาน\n2. บันทึกบุคคลที่สามหรือผู้พบเห็น\n3. หลีกเลี่ยงการตอบโต้ด้วยถ้อยคำรุนแรง\n4. ปรึกษาทนายหรือพนักงานสอบสวน")
        st.markdown("**สำหรับผู้โพสต์/ผู้ถูกกล่าวหา**\n\n1. หยุดเผยแพร่หรือแชร์ซ้ำ\n2. ตรวจสอบข้อเท็จจริงและบริบท\n3. ปรึกษาทนายก่อนชี้แจงหรือลบหลักฐาน")
    with tab4:
        for section, data in LEGAL_SECTIONS_DB.items():
            with st.expander(f"{section}: {data['title']}"):
                st.write(data["summary"])
                for element in data["elements"]:
                    st.write(f"- {element}")

    st.divider()
    st.subheader("Active Learning")
    st.write(f"สะสมข้อความใหม่ได้แล้ว {collection_count()} รายการ (พร้อมสำหรับ Retrain เมื่อครบ 50 รายการ)")
    correct_label = st.selectbox("ยืนยัน Class ที่ถูกต้อง", list(LABELS.values()))
    if st.button("ส่งข้อเสนอแนะ / บันทึกข้อมูลเพื่อฝึกฝนโมเดล"):
        append_feedback(cleaned, predicted, correct_label)
        st.success("บันทึกข้อมูลลง train_dataset_augmented.csv แล้ว")
else:
    st.info("กรอกข้อความทางแถบด้านซ้าย แล้วกด เริ่มวิเคราะห์ข้อความ")
