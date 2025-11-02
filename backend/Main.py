import streamlit as st
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="AI Assistant", layout="wide")

st.sidebar.header("🔗 URL Input")

# Persistent history using Streamlit session state
if "history" not in st.session_state:
    st.session_state.history = []

# URL input field
url = st.sidebar.text_input("Enter URL...", placeholder="https://example.com")

# Load URL button
if st.sidebar.button("Load"):
    if url:
        st.session_state.history.append(url)
        st.session_state["current_url"] = url
        st.session_state["url_loaded"] = True
    else:
        st.sidebar.warning("Please enter a URL.")

# Display Recent History
st.sidebar.markdown("### 🕒 Recent History")
for item in st.session_state.history[-5:][::-1]:
    st.sidebar.write(item)


# Main Area - Chat Interface

st.title("🤖 AI Assistant")

# If a URL is loaded
if st.session_state.get("url_loaded"):
    current_url = st.session_state.get("current_url")
    st.markdown(f"**URL loaded:** {current_url}")

    # Try fetching content
    try:
        response = requests.get(current_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text_content = " ".join([p.get_text(strip=True) for p in soup.find_all("p")])
        st.session_state["page_content"] = text_content
        st.success("✅ Page content loaded successfully.")
    except Exception as e:
        st.error(f"Error loading URL: {e}")

# Chat section
user_question = st.chat_input("Ask a question about the URL content...")

if user_question:
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        if "page_content" not in st.session_state:
            st.write("No URL content loaded yet. Please load a URL first.")
        else:
            # For now, mock response (you can integrate AI later)
            st.markdown("This is a mock response. To get real answers:")
            st.markdown(f"**Your question was:** “{user_question}”")

