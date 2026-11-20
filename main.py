import streamlit as st

st.title("AI Document Assistant")
st.success("Application started successfully!")

import extractfiles as ex
import generatecontent as gc
from pypdf import PdfReader
from docx import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_reranker():
    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )



st.title ("AI Document Assitant")
uploaded_file = st.file_uploader ("Upload Your Document",type = ["pdf","docx","txt"])
text=""
reader=""

if uploaded_file:

 MAX_FILE_SIZE = 10 * 1024 * 1024

 if uploaded_file.size > MAX_FILE_SIZE:

        st.error("File size must be less than 10 MB.")

        st.stop()  


 file_name=uploaded_file.name
 st.success(f"Uploaded File:{file_name}")

 if file_name.lower().endswith(".pdf"):
  text,reader=ex.extract_pdf(uploaded_file)

 elif file_name.lower().endswith(".docx"):
  text=ex.extract_docx(uploaded_file)
 
 elif file_name.lower().endswith(".txt"):
  text=ex.extract_txt(uploaded_file)
  
 else:
   text=""

 st.subheader ("Contents !")
 st.text_area ("Document Content",text,height=500) 
 
 st.subheader("Document Details !")
 col1,col2,col3=st.columns(3)
 col1.metric ("Characters :",len(text))
 col2.metric ("words :",len(text.split()))
 col3.metric("Pages :",len(reader.pages)if file_name.endswith(".pdf")else "N/A" )

 cleaned_text=ex.clean_text(text)
 st.write("Original characters:", len(text))
 st.write("Cleaned characters:", len(cleaned_text))

 chunks=ex.create_chunks(cleaned_text)
 st.write ("Number of Chunks:",len(chunks))
 
 #for i,chunk in enumerate(chunks):
  #with st.expander(f"Chunk {i+1}"):
   #st.write (chunk)

 from sentence_transformers import SentenceTransformer

 embedding_model = load_embedding_model()
 embeddings = embedding_model.encode(
    chunks,
    convert_to_numpy=True, normalize_embeddings=True)
 st.write ("Embedding Shape:",embeddings.shape)

 import faiss
 dimension=embeddings.shape[1]
 index=faiss.IndexFlatIP(dimension)
 index.add(embeddings)
 
 st.write ("Vector Database Created !")

 user_question=st.text_input ("Ask a question about your document !")

 tokenized_chunks = [ex.tokenize(chunk) for chunk in chunks]
 bm25=BM25Okapi(tokenized_chunks)  
 reranker = load_reranker()

 if user_question:
  #question_embedding=embedding_model.encode([user_question],convert_to_numpy=True,normalize_embeddings=True)
  #k=min (6,len(chunks))
  #distances,indices=index.search(question_embedding,k=k) 

  #retrieved_chunks = []
  #for idx in indices[0]:
    #retrieved_chunks.append(chunks[idx])

  results =ex.hybrid_search (embedding_model,index,bm25,chunks,user_question,top_k=5)  
  
  final_results=ex.rerank_results(user_question,results,reranker,top_k=5)
  
  retrieved_chunks = [
result["chunk"]for result in final_results
]
  
  answer=gc.generate_answer (user_question,retrieved_chunks)  
  st.subheader("Final Answer")

  st.write(answer)
