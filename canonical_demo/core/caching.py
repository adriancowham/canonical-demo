import streamlit as st
from streamlit.runtime.caching.hashing import HashFuncsDict

import canonical_demo.core.parsing as parsing
import canonical_demo.core.chunking as chunking
import canonical_demo.core.embedding as embedding
from canonical_demo.core.parsing import File


def file_hash_func(file: File) -> str:
    """Get a unique hash for a file"""
    return file.id


@st.cache_data(show_spinner=False)
def bootstrap_caching():
    """Patch module functions with caching"""

    # Get all substypes of File from module
    file_subtypes = [
        cls
        for cls in vars(parsing).values()
        if isinstance(cls, type) and issubclass(cls, File) and cls != File
    ]
    file_hash_funcs: HashFuncsDict = {cls: file_hash_func for cls in file_subtypes}

    parsing.read_file = st.cache_data(show_spinner=False)(parsing.read_file)
    chunking.chunk_file = st.cache_data(show_spinner=False, hash_funcs=file_hash_funcs)(
        chunking.chunk_file
    )
    embedding.embed_files = st.cache_data(
        show_spinner=False, hash_funcs=file_hash_funcs
    )(embedding.embed_files)
