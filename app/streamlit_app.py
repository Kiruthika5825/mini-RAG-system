import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG App", layout="wide")

left, right = st.columns([1, 2])

# ============================================
# LEFT PANEL — FILE & URL LOADING
# ============================================

with left:
    st.header("Sources")

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["pdf", "txt", "docx", "png", "jpg", "jpeg"]
    )

    if uploaded_file and st.button("Upload"):
        with st.spinner("Processing & embedding..."):
            try:
                response = requests.post(
                    f"{API_URL}/load/upload",
                    files={"file": uploaded_file}
                )
                resp_json = response.json()
            except Exception as e:
                st.error(f"Error uploading file: {e}")
                resp_json = None

        if resp_json:
            st.success(f"File indexed ✔ — {resp_json.get('chunks_indexed', 0)} chunks added")
            st.json(resp_json)

    url = st.text_input("Enter URL here")

    if st.button("Load URL") and url.strip():
        with st.spinner("Scraping + storing in DB..."):
            try:
                response = requests.post(
                    f"{API_URL}/load/url",
                    json={"url": url}
                )
                resp_json = response.json()
            except Exception as e:
                st.error(f"Error loading URL: {e}")
                resp_json = None

        if resp_json:
            st.success(f"URL indexed ✔ — {resp_json.get('chunks_indexed', 0)} chunks added")
            st.json(resp_json)


# ============================================
# RIGHT PANEL — RAG QUERY + SCORE DISPLAY
# ============================================

with right:
    st.header("Ask AI")

    question = st.text_input("Ask a Question")

    if st.button("Get RAG Answer") and question.strip():
        with st.spinner("Searching + Generating..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"question": question, "top_k": 4}
                )
                result = response.json()
            except Exception as e:
                st.error(f"Error contacting backend: {e}")
                result = {}

        if result:
            st.subheader("Answer")
            st.write(result.get("answer", "No answer returned"))

            # === Evaluation Score ===
            score = result.get("evaluation_score", None)
            if score is not None:
                st.subheader("RAG Evaluation Score")
                st.write(f"**{score:.2f}%**")
                st.progress(min(max(score / 100, 0), 1))
            else:
                st.warning("Backend did not return 'evaluation_score'.")
