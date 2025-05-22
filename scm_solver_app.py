import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import openai
import json
from solvers import solve_break_even, solve_transportation
import pandas as pd
import matplotlib.pyplot as plt

openai.api_key = st.secrets["OPENAI_API_KEY"]
reader = easyocr.Reader(['en', 'vi'], gpu=False)

st.title("📸 AI Giải Bài Tập LSCM Từ Ảnh")

uploaded_image = st.file_uploader("Upload ảnh đề bài", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Ảnh đề bài", use_column_width=True)

    with st.spinner("🔍 Đang trích xuất văn bản từ ảnh..."):
        result = reader.readtext(np.array(image), detail=0)
        extracted_text = "\n".join(result)
        st.subheader("📄 Văn bản OCR:")
        st.code(extracted_text)

    if st.button("🧠 Phân tích & bóc tách tham số"):
        prompt = f"""
Bạn là một trợ lý toán học. Dưới đây là đề bài:
\"\"\"{extracted_text}\"\"\"
Hãy xác định loại bài toán và trích xuất các tham số dưới dạng JSON.
"""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Bạn là chuyên gia giải bài toán LSCM"},
                {"role": "user", "content": prompt}
            ]
        )
        result_text = response.choices[0].message["content"]
        st.subheader("🧾 Tham số trích xuất:")
        st.code(result_text)

        try:
            parsed_data = json.loads(result_text)
            ptype = parsed_data["problem_type"].lower()

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
                fig.colorbar(c)
                st.pyplot(fig)

                st.info(result["explanation"])

        except Exception as e:
            st.error(f"❌ Không thể đọc dữ liệu JSON: {e}")
