
import streamlit as st

import extractfiles as ex
import generatecontent as gc

from rank_bm25 import BM25Okapi


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("AI Document Assistant")


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )


# --------------------------------------------------
# LOAD RERANKER MODEL
# --------------------------------------------------

@st.cache_resource
def load_reranker():

    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Your Document",
    type=["pdf", "docx", "txt"]
)


if uploaded_file:

    MAX_FILE_SIZE = 10 * 1024 * 1024

    if uploaded_file.size > MAX_FILE_SIZE:

        st.error("File size must be less than 10 MB.")

        st.stop()


    file_name = uploaded_file.name

    st.success(f"Uploaded File: {file_name}")


    # --------------------------------------------------
    # EXTRACT DOCUMENT CONTENT
    # --------------------------------------------------

    try:

        if file_name.lower().endswith(".pdf"):

            text, reader = ex.extract_pdf(uploaded_file)

            page_count = len(reader.pages)

        elif file_name.lower().endswith(".docx"):

            text = ex.extract_docx(uploaded_file)

            page_count = "N/A"

        elif file_name.lower().endswith(".txt"):

            text = ex.extract_txt(uploaded_file)

            page_count = "N/A"

        else:

            st.error("Unsupported file format.")

            st.stop()


    except Exception as error:

        st.error(f"Document extraction failed: {error}")

        st.stop()


    if not text.strip():

        st.error("No readable text found in the document.")

        st.stop()


    # --------------------------------------------------
    # DISPLAY DOCUMENT CONTENT
    # --------------------------------------------------

    st.subheader("Document Contents")

    st.text_area(
        "Document Content",
        text,
        height=400
    )


    # --------------------------------------------------
    # DOCUMENT DETAILS
    # --------------------------------------------------

    st.subheader("Document Details")

    col1, col2, col3 = st.columns(3)

    col1.metric("Characters", len(text))

    col2.metric("Words", len(text.split()))

    col3.metric("Pages", page_count)


    # --------------------------------------------------
    # CLEAN DOCUMENT
    # --------------------------------------------------

    cleaned_text = ex.clean_text(text)

    st.write("Original characters:", len(text))

    st.write("Cleaned characters:", len(cleaned_text))


    # --------------------------------------------------
    # CREATE CHUNKS
    # --------------------------------------------------

    chunks = ex.create_chunks(cleaned_text)

    if not chunks:

        st.error("Unable to create document chunks.")

        st.stop()

    st.write("Number of Chunks:", len(chunks))


    # --------------------------------------------------
    # GENERATE EMBEDDINGS
    # --------------------------------------------------

    with st.spinner("Loading embedding model..."):

        embedding_model = load_embedding_model()

    with st.spinner("Generating document embeddings..."):

        embeddings = embedding_model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    st.write("Embedding Shape:", embeddings.shape)


    # --------------------------------------------------
    # CREATE FAISS VECTOR DATABASE
    # --------------------------------------------------

    import faiss

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    st.success("Vector Database Created!")


    # --------------------------------------------------
    # CREATE BM25 INDEX
    # --------------------------------------------------

    tokenized_chunks = [
        ex.tokenize(chunk)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)


    # --------------------------------------------------
    # LOAD RERANKER
    # --------------------------------------------------

    with st.spinner("Loading reranking model..."):

        reranker = load_reranker()


    # --------------------------------------------------
    # USER QUESTION
    # --------------------------------------------------

    user_question = st.text_input(
        "Ask a question about your document!"
    )


    if user_question:

        # Hybrid Search

        results = ex.hybrid_search(
            embedding_model,
            index,
            bm25,
            chunks,
            user_question,
            top_k=5
        )


        # Reranking

        final_results = ex.rerank_results(
            user_question,
            results,
            reranker,
            top_k=5
        )


        # Retrieve Relevant Chunks

        retrieved_chunks = [
            result["chunk"]
            for result in final_results
        ]


        # Generate Final Answer

        with st.spinner("Generating answer..."):

            answer = gc.generate_answer(
                user_question,
                retrieved_chunks
            )


        st.subheader("Final Answer")

        st.write(answer)
