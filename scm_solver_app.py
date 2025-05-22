import streamlit as st
from PIL import Image
import numpy as np
import pytesseract
import openai
import json
import pandas as pd
import matplotlib.pyplot as plt
import re
from streamlit_pasteimage import paste_image

from solvers import solve_break_even, solve_transportation

# Đặt API Key từ secrets
openai.api_key = st.secrets.get("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", "")
if not openai.api_key:
    st.error("❌ OPENAI_API_KEY chưa được thiết lập trong secrets. Vui lòng cấu hình trong Streamlit Cloud.")
    st.stop()

# Cấu hình Streamlit
st.set_page_config(page_title="AI Giải Bài Tập LSCM", layout="wide")
st.title("📸 AI Giải Bài Tập LSCM Từ Ảnh")

# --- Giao diện Upload / Paste ---
tab1, tab2 = st.tabs(["📤 Upload ảnh", "📋 Dán ảnh (Ctrl+V)"])
with tab1:
    uploaded_image = st.file_uploader("Chọn ảnh đề bài", type=["jpg", "jpeg", "png"])
with tab2:
    pasted_image = paste_image()

image = pasted_image or uploaded_image

# --- Xử lý ảnh và OCR ---
if image:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    st.image(image, caption="Ảnh đề bài", use_column_width=True)

    with st.spinner("🔍 Đang trích xuất văn bản từ ảnh..."):
        extracted_text = pytesseract.image_to_string(image, lang="eng+vie")
        st.subheader("📄 Văn bản OCR:")
        st.code(extracted_text)

    # --- Gửi đề bài đến GPT ---
    if st.button("🧠 Phân tích & bóc tách tham số"):
        prompt = f"""
Bạn là một trợ lý toán học. Dưới đây là đề bài:
\"\"\"{extracted_text}\"\"\"
Hãy xác định loại bài toán (ví dụ: break-even, transportation, inventory, assignment...) và trích xuất các tham số cần thiết để giải bài toán đó dưới dạng JSON hợp lệ. 
Đáp án phải nằm trong khối mã JSON như sau:
```json
{{ "problem_type": ..., "fixed_cost": ..., "variable_cost": ..., ... }}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia giải bài toán LSCM"},
            {"role": "user", "content": prompt}
        ]
    )
    result_text = response.choices[0].message["content"]
    st.subheader("🧾 Tham số trích xuất:")
    st.code(result_text)

    # --- Tách và đọc JSON ---
    try:
        json_match = re.search(r"```json(.*?)```", result_text, re.DOTALL)
        json_content = json_match.group(1).strip() if json_match else result_text
        parsed_data = json.loads(json_content)

        ptype = parsed_data["problem_type"].lower()

        # --- Giải bài toán ---
        if ptype == "break-even":
            result = solve_break_even(parsed_data)
            st.metric("✅ Break-even Point", result["break_even_point"])
            st.info(result["explanation"])

        elif ptype == "transportation":
            result = solve_transportation(parsed_data)
            st.metric("✅ Tổng chi phí tối ưu", round(result["total_cost"], 2))

            st.subheader("📊 Phân phối hàng hóa:")
            df = pd.DataFrame(result["solution_matrix"])
            st.dataframe(df)

            fig, ax = plt.subplots()
            c = ax.imshow(result["solution_matrix"], cmap='Blues')
            ax.set_xticks(np.arange(df.shape[1]))
            ax.set_yticks(np.arange(df.shape[0]))
            ax.set_xticklabels([f"B{i+1}" for i in range(df.shape[1])])
            ax.set_yticklabels([f"A{i+1}" for i in range(df.shape[0])])
            plt.colorbar(c)
            st.pyplot(fig)

            st.info(result["explanation"])

        else:
            st.warning(f"📌 Chưa hỗ trợ loại bài toán: {ptype}")

    except Exception as e:
        st.error(f"❌ Không thể đọc dữ liệu JSON: {e}")
        st.code(result_text)
