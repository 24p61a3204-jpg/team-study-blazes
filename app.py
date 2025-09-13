import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="StudyMate", page_icon="📚", layout="wide")
st.title("📚 StudyMate: AI-Powered PDF Q&A System")

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    with st.spinner("Uploading & processing PDF..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{BACKEND_URL}/upload", files={"file": uploaded_file})
        if response.status_code == 200:
            data = response.json()
            st.session_state.doc_id = data["doc_id"]
            st.success(f"✅ Uploaded {data['filename']} with {data['pages']} pages & {data['chunks']} chunks.")
        else:
            st.error(f"Upload failed: {response.text}")

if st.session_state.doc_id:
    question = st.text_input("Ask a question about the PDF:")
    if st.button("Get Answer") and question.strip():
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{BACKEND_URL}/ask/{st.session_state.doc_id}",
                data={"question": question, "top_k": 4}
            )
            if response.status_code == 200:
                data = response.json()
                st.subheader("Answer:")
                st.write(data["answer"])
                with st.expander("View Source Chunks"):
                    for i, chunk in enumerate(data["source_chunks"], 1):
                        st.markdown(f"**Chunk {i}:** {chunk['page_content']}")
            else:
                st.error(f"Error: {response.text}")
