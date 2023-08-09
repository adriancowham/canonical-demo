import os

import streamlit as st

from canonical_demo.components.sidebar import sidebar
from canonical_demo.core.caching import bootstrap_caching
from canonical_demo.core.chunking import chunk_file
from canonical_demo.core.embedding import embed_files
from canonical_demo.core.parsing import read_file
from canonical_demo.core.qa import query_folder
from canonical_demo.ui import (
    display_file_read_error,
    is_file_valid,
    is_open_ai_key_valid,
    is_query_valid,
    wrap_doc_in_html,
)

EMBEDDING = "openai"
VECTOR_STORE = "faiss"
MODEL = "openai"

# For testing
# EMBEDDING, VECTOR_STORE, MODEL = ["debug"] * 3

st.set_page_config(page_title="Canonical Demo - Let's Talk...", page_icon=None, layout="wide")
st.header("Canonical Demo")

# Enable caching for expensive functions
bootstrap_caching()
openai_api_key = os.environ["OPENAI_API_KEY"]

with open('./resources/progit.pdf', 'rb') as uploaded_file:
  try:
      file = read_file(uploaded_file)
  except Exception as e:
      display_file_read_error(e)

chunked_file = chunk_file(file, chunk_size=300, chunk_overlap=0)

if not is_file_valid(file):
    st.stop()

if not is_open_ai_key_valid(openai_api_key):
    st.stop()


with st.spinner("Indexing document...this may take a while."):
    folder_index = embed_files(
        files=[chunked_file],
        embedding=EMBEDDING,
        vector_store=VECTOR_STORE,
        openai_api_key=openai_api_key,
    )

with st.form(key="qa_form"):
    query = st.text_area("Ask a question and, Let's Talk...")
    submit = st.form_submit_button("Submit")


return_all_chunks = True
show_full_doc = True


if show_full_doc:
    with st.expander("Document"):
        # Hack to get around st.markdown rendering LaTeX
        st.markdown(f"<p>{wrap_doc_in_html(file.docs)}</p>", unsafe_allow_html=True)


if submit:
    if not is_query_valid(query):
        st.stop()

    # Output Columns
    answer_col, sources_col = st.columns(2)

    result = query_folder(
        folder_index=folder_index,
        query=query,
        return_all=return_all_chunks,
        model=MODEL,
        openai_api_key=openai_api_key,
        temperature=0,
    )

    with answer_col:
        st.markdown("#### Answer")
        st.markdown(result.answer)

    with sources_col:
        st.markdown("#### Sources")
        for source in result.sources:
            st.markdown(source.page_content)
            st.markdown(source.metadata["source"])
            st.markdown("---")
