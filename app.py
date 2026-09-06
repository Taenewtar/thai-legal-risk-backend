import csv
import hashlib
import random
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from pythainlp.tag import pos_tag
from pythainlp.tokenize import word_tokenize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_PATH = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_PATH
LABELS = {0: "ทั่วไป/ติชม", 1: "ดูหมิ่น", 2: "หมิ่นประมาท ม.326"}

LEGAL_SECTIONS_DB = {
    "มาตรา 393": {"title": "ดูหมิ่น", "summary": "ถ้อยคำดูถูก เหยียดหยาม หรือทำให้อับอาย", "elements": ["มีการดูถูกหรือเหยียดหยาม", "ต้องพิจารณาบริบทและเจตนา"]},
    "มาตรา 326": {"title": "หมิ่นประมาททั่วไป", "summary": "ใส่ความผู้อื่นต่อบุคคลที่สามจนมีแนวโน้มทำให้เสียชื่อเสียง", "elements": ["มีผู้ถูกกล่าวถึง", "มีบุคคลที่สามรับรู้", "ข้อความอาจทำให้เสียชื่อเสียง ถูกดูหมิ่น หรือถูกเกลียดชัง"]},
    "มาตรา 328": {"title": "หมิ่นประมาทโดยการโฆษณา", "summary": "เผยแพร่ข้อความหรือสื่อให้บุคคลทั่วไปเข้าถึงได้", "elements": ["เผยแพร่ต่อสาธารณะ", "มีลักษณะใส่ความตามมาตรา 326"]},
    "มาตรา 327": {"title": "หมิ่นประมาทผู้ตาย", "summary": "กล่าวถึงผู้ตายจนกระทบชื่อเสียงของบุคคลในครอบครัวตามกฎหมาย", "elements": ["ผู้ถูกกล่าวถึงเป็นผู้ตาย", "กระทบชื่อเสียงหรือความรู้สึกของบุคคลที่กฎหมายคุ้มครอง"]},
    "มาตรา 329": {"title": "แสดงความคิดเห็นโดยสุจริต", "summary": "การติชมโดยสุจริตหรือปกป้องสิทธิโดยชอบอาจเป็นข้อพิจารณา", "elements": ["มีความสุจริต", "ใช้ถ้อยคำเท่าที่จำเป็น", "เกี่ยวข้องกับการปกป้องสิทธิหรือประโยชน์สาธารณะ"]},
}

LEGAL_ADVICE_DB = {
    "SEC_326": {
        "sections": "มาตรา 326",
        "rec_victim": [
            "1. เก็บข้อความฉบับเต็ม พร้อมชื่อบัญชี ลิงก์ และภาพหน้าจอที่เห็นวันเวลา",
            "2. บันทึกว่ามีบุคคลที่สามคนใดเห็นหรือได้รับข้อความ และเก็บพยานที่เกี่ยวข้อง",
            "3. หลีกเลี่ยงการตอบโต้ด้วยถ้อยคำรุนแรงหรือเผยแพร่ซ้ำ เพราะอาจสร้างประเด็นโต้กลับ",
            "4. นำหลักฐานไปปรึกษาพนักงานสอบสวนหรือทนาย เพื่อประเมินองค์ประกอบมาตรา 326 และกำหนดแนวทางดำเนินคดี",
        ],
        "rec_poster": [
            "1. หยุดเผยแพร่หรือลบการแชร์ต่อ และเก็บสำเนาข้อความกับบริบทไว้เพื่อชี้แจง",
            "2. ตรวจสอบข้อเท็จจริง แหล่งที่มา เจตนา และบุคคลที่สามที่ได้รับข้อความก่อนให้ถ้อยคำ",
            "3. ปรึกษาทนายก่อนรับสารภาพ ชี้แจง หรือติดต่อคู่กรณี และพิจารณาแก้ไขหรือลงข้อความชี้แจงอย่างเหมาะสม",
        ],
    },
    "SEC_328": {
        "sections": "มาตรา 328",
        "rec_victim": [
            "1. เก็บหลักฐานการเผยแพร่ต่อสาธารณะทั้งหมด รวม URL ภาพหน้าจอ วันเวลา ยอดเข้าถึง และการแชร์",
            "2. บันทึกช่องทางเผยแพร่และรายชื่อพยานหรือผู้ที่พบเห็นโพสต์ โดยอย่าตัดต่อหลักฐานต้นฉบับ",
            "3. ใช้ช่องทางรายงานหรือลบเนื้อหาของแพลตฟอร์มเท่าที่จำเป็น และหลีกเลี่ยงการโพสต์ซ้ำเพื่อโต้ตอบ",
            "4. ปรึกษาพนักงานสอบสวนหรือทนายเพื่อประเมินการเผยแพร่ต่อสาธารณะตามมาตรา 328 และความเสียหายที่เกิดขึ้น",
        ],
        "rec_poster": [
            "1. หยุดโพสต์ แชร์ หรือเผยแพร่ซ้ำ และเก็บหลักฐานต้นฉบับรวมถึงบริบททั้งหมดไว้ก่อนแก้ไข",
            "2. ตรวจสอบข้อเท็จจริง ขอบเขตผู้รับสาร และเจตนาการเผยแพร่ เพราะการโฆษณาอาจมีผลต่อการพิจารณา",
            "3. ปรึกษาทนายก่อนลบหรือชี้แจงอย่างเป็นทางการ และพิจารณาการแก้ไขเยียวยาที่ไม่เพิ่มความเสียหาย",
        ],
    },
    "SEC_393": {
        "sections": "มาตรา 393",
        "rec_victim": [
            "1. เก็บข้อความ ภาพหน้าจอ วันเวลา สถานที่ และพยานที่อยู่ในเหตุการณ์",
            "2. หลีกเลี่ยงการโต้เถียงหรือเผยแพร่ข้อความตอบโต้ที่อาจเป็นความผิดอีกกรณี",
            "3. ปรึกษาพนักงานสอบสวนหรือทนายเพื่อประเมินลักษณะการดูหมิ่นและพฤติการณ์แวดล้อม",
            "4. เก็บหลักฐานความเสียหายและการติดต่อไกล่เกลี่ยไว้เป็นระบบ",
        ],
        "rec_poster": [
            "1. หยุดใช้ถ้อยคำดังกล่าวและอย่าเผยแพร่ซ้ำ",
            "2. เก็บบริบททั้งหมดและหลีกเลี่ยงการลบหรือแก้ไขหลักฐานโดยไม่ปรึกษาผู้เชี่ยวชาญ",
            "3. ขอคำปรึกษาทนายก่อนชี้แจงหรือเจรจากับคู่กรณี",
        ],
    },
    "SEC_329": {
        "sections": "มาตรา 329",
        "rec_victim": ["1. ตรวจสอบว่าข้อความเป็นการติชมโดยสุจริตหรือเป็นการกล่าวข้อเท็จจริงที่เกินจำเป็น", "2. เก็บบริบทและหลักฐานเพื่อประเมินความเสียหาย", "3. ปรึกษาทนายหากมีการเผยแพร่ต่อบุคคลที่สาม", "4. ใช้ช่องทางแก้ไขหรือติดต่อผู้เผยแพร่โดยไม่เพิ่มความขัดแย้ง"],
        "rec_poster": ["1. ตรวจสอบข้อเท็จจริงและแหล่งที่มาของข้อมูล", "2. จำกัดถ้อยคำให้สุจริต จำเป็น และเกี่ยวข้องกับประโยชน์สาธารณะ", "3. เก็บหลักฐานประกอบเจตนาและข้อเท็จจริงไว้"],
    },
}

defamation_keywords = [
    "ขายยา", "ขายยาบ้า", "ค้ายา", "แอบขายยา", "เอเย่นต์ยา", "ค้ายาเสพติด", "ฟอกเงิน", "อุ้มฆ่า", "บ่อน", "โต๊ะบอล", "เจ้ามือหวย", "ซูเอี๋ย",
    "โกง", "โกงเงิน", "ทุจริต", "ยักยอก", "แอบยักยอก", "รับสินบน", "คอรัปชั่น", "คอร์รัปชัน", "กินส่วนต่าง", "แดกงบ", "ต้มตุ๋น", "ตบทรัพย์", "รีดไถ", "ไถเงิน", "ยัดข้อหา", "ฉ้อโกง", "หลอกลวง", "ขโมย",
    "เป็นชู้", "แอบเป็นชู้", "ชู้", "เมียน้อย", "เมียเก็บ", "เต้าไต่", "ชู้สาว", "มั่ว", "ขายตัว", "มั่วผู้ชาย", "มั่วผู้หญิง", "นอกใจ", "มีชู้", "ค้าประเวณี", "ขายบริการ", "อีตัว", "กะหรี่", "แมงดา",
]
insult_keywords = [
    "เหี้ย", "ควย", "ส้นตีน", "สถุล", "ขยะสังคม", "เศษเดน", "เฮงซวย", "ตอแหล", "ชาติชั่ว", "ระยำ", "อัปสรี", "ชั่วช้า", "เลวทราม", "ต่ำช้า", "หน้าด้าน", "โง่", "ปัญญาอ่อน", "ควาย", "สารเลว", "สวะ", "เดนสังคม", "บัดซบ", "ถ่อย", "กาก", "จัญไร", "ไร้ค่า", "หน้าตัวเมีย",
]
RISKY_PREFIXES = {"แอบ", "กำลัง", "ชอบ", "เคย", "คิดจะ", "พยายาม"}
EXAMPLE_TEXTS = [
    "สั่งอาหารไปสองชั่วโมงแล้ว ยังไม่ได้รับอาหารเลย บริการช้ามาก",
    "สันดานโกงแบบนี้ อย่าไปทำธุรกิจด้วยเด็ดขาด",
    "หัวหน้าฝ่ายบัญชีบริษัทนี้ยักยอกเงินบริษัทไปใช้ส่วนตัว",
]
COLLECTED_PATH = PROJECT_PATH / "collected_dataset.csv"
AUGMENTED_PATH = PROJECT_PATH / "train_dataset_augmented.csv"
COLLECTED_FIELDS = ["timestamp", "text", "predicted_class", "confidence", "extracted_keywords"]
AUGMENTED_FIELDS = ["timestamp", "user_text", "predicted_sections", "correct_label"]

print(f"กำลังโหลดโมเดลจาก {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
model.eval()
print("โหลดโมเดลสำเร็จ")


def tokens_for(text):
    return word_tokenize(str(text), engine="newmm")


def tagged_for(text):
    tokens = tokens_for(text)
    try:
        tagged = pos_tag(tokens, corpus="orchid", tagset="universal")
    except Exception:
        tagged = pos_tag(tokens, corpus="orchid")
    return tokens, tagged


def pos_risk_terms(text):
    _, tagged = tagged_for(text)
    result = []
    for token, tag in tagged:
        tag_text = str(tag).upper()
        if tag_text in {"VERB", "ADJ"} or tag_text.startswith(("V", "A", "JJ")):
            if len(token.strip()) > 1:
                result.append(token)
    return list(dict.fromkeys(result))


def risky_phrases(text):
    tokens, tagged = tagged_for(text)
    result = []
    for index, (_, tag) in enumerate(tagged):
        tag_text = str(tag).upper()
        if tag_text not in {"VERB", "ADJ"} and not tag_text.startswith(("V", "A", "JJ")):
            continue
        start = index - 1 if index and tokens[index - 1] in RISKY_PREFIXES else index
        phrase = "".join(tokens[start:min(len(tokens), index + 3)]).strip()
        if len(phrase) > 2:
            result.append(phrase)
    return list(dict.fromkeys(result))


def find_keywords(text):
    keywords = [word for word in sorted(set(defamation_keywords + insult_keywords), key=len, reverse=True) if word in text]
    if not keywords:
        keywords = risky_phrases(text) + pos_risk_terms(text)
    return list(dict.fromkeys(keywords))


def model_analysis(text):
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


def section_report(predicted, confidence):
    selected = ["มาตรา 393"] if predicted == 1 else ["มาตรา 326", "มาตรา 328"] if predicted == 2 else ["มาตรา 329"]
    lines = [f"**มาตราที่ควรตรวจต่อ:** {', '.join(selected)}", f"**ความมั่นใจ:** {confidence * 100:.2f}%", ""]
    for section, data in LEGAL_SECTIONS_DB.items():
        marker = "เกี่ยวข้องกับผลคัดกรอง" if section in selected else "ข้อมูลประกอบ"
        lines.append(f"#### {section}: {data['title']} ({marker})")
        lines.append(data["summary"])
        lines.extend(f"- {item}" for item in data["elements"])
    return "\n".join(lines)


def detected_advice_keys(predicted, text=""):
    if predicted == 1:
        return ["SEC_393"]
    if predicted == 2:
        public_markers = ("โพสต์", "เฟซบุ๊ก", "facebook", "เพจ", "ออนไลน์", "อินเทอร์เน็ต", "สาธารณะ", "แชร์", "กลุ่ม")
        return ["SEC_328"] if any(marker in text.lower() for marker in public_markers) else ["SEC_326"]
    return ["SEC_329"]


def build_tab3_content(predicted, text=""):
    keys = detected_advice_keys(predicted, text)
    victim = []
    poster = []
    for key in keys:
        victim.extend(LEGAL_ADVICE_DB[key]["rec_victim"])
        poster.extend(LEGAL_ADVICE_DB[key]["rec_poster"])
    sections = ", ".join(LEGAL_ADVICE_DB[key]["sections"] for key in keys)
    victim_text = "\n".join(victim)
    poster_text = "\n".join(poster)
    return (
        f"## แนวทางรับมือสำหรับ {sections}\n\n"
        "### สำหรับผู้เสียหาย\n"
        f"{victim_text}\n\n"
        "### สำหรับผู้โพสต์/ผู้ถูกกล่าวหา\n"
        f"{poster_text}\n\n"
        "> แนวทางนี้เป็นข้อมูลเบื้องต้น ควรตรวจข้อเท็จจริงและปรึกษาผู้ประกอบวิชาชีพกฎหมาย"
    )


def predicted_sections(predicted):
    if predicted == 1:
        return "มาตรา 393"
    if predicted == 2:
        return "มาตรา 326, มาตรา 328"
    return "มาตรา 329"


def write_collection(text, predicted, confidence, keywords):
    exists = COLLECTED_PATH.exists() and COLLECTED_PATH.stat().st_size > 0
    with COLLECTED_PATH.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLLECTED_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({"timestamp": datetime.now(timezone.utc).isoformat(), "text": text, "predicted_class": predicted, "confidence": f"{confidence:.6f}", "extracted_keywords": "|".join(keywords)})


def collected_status():
    if not COLLECTED_PATH.exists():
        count = 0
    else:
        with COLLECTED_PATH.open(encoding="utf-8-sig", newline="") as file:
            count = sum(1 for _ in csv.DictReader(file))
    suffix = "พร้อมสำหรับ Retrain" if count >= 50 else "พร้อมสำหรับ Retrain เมื่อครบ 50 รายการ"
    return f"สะสมข้อความใหม่ได้แล้ว {count} รายการ ({suffix})"


def analyze_legal_text(text):
    try:
        cleaned, predicted, confidence, probabilities, keywords = model_analysis(text)
    except ValueError as error:
        return str(error), "", "", "", collected_status(), None
    write_collection(cleaned, predicted, confidence, keywords)
    probability_text = "\n".join(f"- {LABELS.get(index, f'คลาส {index}')}: {float(value) * 100:.2f}%" for index, value in enumerate(probabilities))
    keyword_text = "\n".join(f"- `{keyword}`" for keyword in keywords) if keywords else "ไม่พบคำในคลัง จึงไม่พบคำกริยาหรือคำคุณศัพท์ที่ชัดเจนจาก POS Tagging"
    overview = f"## Criminal Legal Risk Engine\n\n**ผลคัดกรอง:** {LABELS[predicted]}\n\n### ความน่าจะเป็น\n{probability_text}\n\nผลนี้เป็นการคัดกรองเบื้องต้น ไม่ใช่คำวินิจฉัยทางกฎหมาย"
    reasoning = f"## เหตุผลและคำเสี่ยง\n\n{keyword_text}\n\n{section_report(predicted, confidence)}"
    tab3_content = build_tab3_content(predicted, cleaned)
    legal = "\n".join(f"- **{key} {data['title']}**: {data['summary']}" for key, data in LEGAL_SECTIONS_DB.items())
    return overview, reasoning, tab3_content, legal, collected_status(), predicted


def analyze_for_ui(text):
    """Backward-compatible alias for older callers."""
    return analyze_legal_text(text)


def save_feedback(text, predicted, correct_label):
    if predicted is None or not str(text).strip() or not correct_label:
        return "กรุณาวิเคราะห์ข้อความและเลือก Class ที่ถูกต้องก่อนบันทึก"
    correct_id = next((index for index, label in LABELS.items() if label == correct_label), None)
    if correct_id is None:
        return "ไม่พบ Class ที่เลือก"
    exists = AUGMENTED_PATH.exists() and AUGMENTED_PATH.stat().st_size > 0
    with AUGMENTED_PATH.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=AUGMENTED_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({"timestamp": datetime.now(timezone.utc).isoformat(), "user_text": str(text).strip(), "predicted_sections": predicted_sections(int(predicted)), "correct_label": correct_id})
    return f"บันทึกข้อมูลเพื่อ Fine-tune แล้ว: Class {correct_id} ลง train_dataset_augmented.csv"


def random_example():
    return random.choice(EXAMPLE_TEXTS)


with gr.Blocks(title="⚖️ ระบบวิเคราะห์ความเสี่ยงทางกฎหมาย (Criminal Legal Risk Engine)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚖️ ระบบวิเคราะห์ความเสี่ยงทางกฎหมาย (Criminal Legal Risk Engine)")
    with gr.Row(equal_height=False):
        with gr.Column(scale=4):
            text_input = gr.Textbox(lines=12, label="ข้อความที่ต้องการตรวจสอบ", placeholder="พิมพ์ข้อความภาษาไทย...")
            with gr.Row():
                analyze_button = gr.Button("🔍 เริ่มวิเคราะห์ข้อความ", variant="primary")
                random_button = gr.Button("🎲 สุ่มข้อความตัวอย่าง")
            gr.Examples(examples=EXAMPLE_TEXTS, inputs=text_input, label="ตัวอย่างข้อความ")
            collection_counter = gr.Markdown(collected_status())
            gr.Markdown("### Active Learning")
            correct_label = gr.Dropdown(choices=list(LABELS.values()), label="ยืนยัน Class ที่ถูกต้อง")
            feedback_button = gr.Button("ส่งข้อเสนอแนะ / ยืนยันผลการวิเคราะห์")
            feedback_status = gr.Markdown()
        with gr.Column(scale=6):
            with gr.Tabs():
                with gr.Tab("📋 สรุปผล & กฎหมายที่เกี่ยวข้อง"):
                    overview_output = gr.Markdown()
                with gr.Tab("🔍 เหตุผล & วิเคราะห์เชิงลึก"):
                    reasoning_output = gr.Markdown()
                with gr.Tab("💡 แนวทางรับมือ"):
                    advice_output = gr.Markdown("กรุณาวิเคราะห์ข้อความเพื่อรับแนวทางรับมือที่เหมาะสม")
                with gr.Tab("📚 เกร็ดความรู้กฎหมาย"):
                    legal_output = gr.Markdown()
    predicted_state = gr.State(value=None)
    random_button.click(random_example, outputs=text_input)
    analyze_button.click(analyze_legal_text, inputs=text_input, outputs=[overview_output, reasoning_output, advice_output, legal_output, collection_counter, predicted_state])
    feedback_button.click(save_feedback, inputs=[text_input, predicted_state, correct_label], outputs=feedback_status)


demo.launch(share=True)
