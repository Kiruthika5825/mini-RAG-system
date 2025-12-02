import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="RAG App", layout="wide")

left, right = st.columns([1,2])

with left:
    st.header("Knowledge Loader")

    uploaded_file = st.file_uploader("Upload file",
                                     type=["pdf","txt","docx","png","jpg","jpeg"])

    if uploaded_file and st.button("Upload"):
        with st.spinner("Processing & embedding..."):
            response = requests.post(
                f"{API_URL}/load/upload",
                files={"file": uploaded_file}
            )
        st.success("File indexed ✔")
        st.json(response.json())

    url = st.text_input("Enter URL here")

    if st.button("Load URL"):
        with st.spinner("Scraping + Storing in DB..."):
            response = requests.post(
                f"{API_URL}/load/url",
                json={"url": url}
            )
        st.success("URL indexed ✔")
        st.json(response.json())

with right:
    st.header("Ask AI from Knowledge Base")

    question = st.text_input("Ask a Question")

    if st.button("Get RAG Answer"):
        with st.spinner("Searching + Generating..."):
            result = requests.post(
                f"{API_URL}/query",
                json={"question": question}
            ).json()

        st.subheader("Answer")
        st.write(result["answer"])
