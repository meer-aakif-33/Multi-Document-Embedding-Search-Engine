"""
streamlit_app.py
A simple Streamlit UI for searching the document embedding engine.
Run with:
    streamlit run streamlit_app.py
Requirements:
    pip install streamlit requests
API must be running at http://127.0.0.1:8000
"""

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/search"

st.set_page_config(page_title="Multi‑Document Search Engine", layout="wide")

st.title("🔍 Multi‑Document Embedding Search Engine")
st.write("Enter a query and view the top‑K semantic matches.")

query = st.text_input("Search query", "machine learning basics")
top_k = st.slider("Top K results", 1, 10, 5)

if st.button("Search"):
    payload = {"query": query, "top_k": top_k}
    with st.spinner("Searching..."):
        try:
            res = requests.post(API_URL, json=payload, timeout=10)
            if res.status_code != 200:
                st.error(f"Error {res.status_code}: {res.text}")
            else:
                data = res.json()
                results = data.get("results", [])
                st.subheader(f"Results for: {query}")
                for r in results:
                    st.markdown(f"### 📄 {r['doc_id']} — Score: **{r['score']:.3f}**")
                    st.write(r['preview'] + "...")
                    with st.expander("Explanation"):
                        st.write("**Keyword Overlap:**", r['explanation']['keyword_overlap'])
                        st.write("**Overlap Ratio:**", r['explanation']['overlap_ratio'])
                        st.write("**Length Normalization:**", r['explanation']['length_norm'])
                    st.markdown("---")
        except Exception as e:
            st.error(f"Request failed: {e}")
