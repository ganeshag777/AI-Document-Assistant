from pypdf import PdfReader 
from docx import Document
from rank_bm25 import BM25Okapi
import numpy as np

def extract_pdf (file):

 reader=PdfReader(file)
 text=""
 for page in reader.pages:
  page_text = page.extract_text()
  if page_text:
   text += page_text +"\n"

 return text,reader 

def extract_docx(file):
 document=Document (file)
 text =""
 for paragraph in document.paragraphs:
  text += paragraph.text +"\n"
  
 return text 

def extract_txt (file):
 return file.read().decode("utf-8")
 
import re
def clean_text(text):
 text=re.sub(r"[  \t]+"," ",text)
 text=re.sub(r"\n+","\n",text)  
 text="\n".join(line.strip() for line in text.splitlines())

 return text.strip()

def create_chunks(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        # Calculate initial end position
        end = min(start + chunk_size, text_length)

        # Move end backward to a space
        if end < text_length:
            space_position = text.rfind(" ", start, end)

            if space_position > start:
                end = space_position

        # Extract chunk
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Stop after the final chunk
        if end >= text_length:
            break

        # Calculate next starting position
        next_start = end - overlap

        # Move next start forward to the next space
        if next_start > 0:
            space_position = text.find(" ", next_start, end)

            if space_position != -1:
                next_start = space_position + 1

        # Safety check: always move forward
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks

def bm25_search(question,bm25,chunks,top_k=3): 
 tokenized_question=tokenize(question)
 scores=bm25.get_scores(tokenized_question)
 top_indices=np.argsort(scores)[::-1][:top_k]
 results=[]
 for idx in top_indices:
  results.append({
           "chunk":chunks[idx],
            "score":float(scores[idx]),
            "index":int(idx)
          })

 return results

import re

stop_words = {
    "a", "an", "the",
    "is", "are", "was", "were",
    "what", "which", "who",
    "and", "or", "of",
    "in", "on", "at",
    "to", "for",
    "its", "it",
    "how", "why"
}

def tokenize(text):

    text = text.lower()

    tokens = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text
    )

    tokens = [
token

                    for token in tokens
                   
 if token not in stop_words
 ]


    return tokens

def normalize_faiss_distances(distances):

    max_distance = max(distances)

    if max_distance == 0:
        return [1.0] * len(distances)

    return [
        1 - (distance / max_distance)
        for distance in distances
    ]

def normalize_bm25_scores(scores):

    max_score = max(scores)

    if max_score == 0:
        return [0.0] * len(scores)

    return [
        score / max_score
        for score in scores
    ]


def hybrid_search(embedding_model,index,bm25,chunks,user_question,top_k=5):
 
 question_embedding=embedding_model.encode([user_question],convert_to_numpy=True,normalize_embeddings=True)
 faiss_distances,faiss_indices=index.search(question_embedding,k=top_k)
 
 query_tokens=tokenize(user_question)
 bm25_scores=bm25.get_scores(query_tokens) 
 bm25_indices=np.argsort(bm25_scores)[::-1][:top_k]


 faiss_distances_list = [float(x)for x in faiss_distances[0]]
 faiss_scores = normalize_faiss_distances(faiss_distances_list)

 bm25_scores_list = [float(bm25_scores[i]) for i in bm25_indices]
 bm25_normalized = normalize_bm25_scores (bm25_scores_list)

 combined={}

 for position,idx in enumerate (faiss_indices[0]):
  idx=int(idx)
  combined[idx]  = {
      "faiss_score":faiss_scores[position],
      "bm25_score":0.0
   }

 for position,idx in enumerate (bm25_indices):
  idx=int(idx)
  if idx not in combined:
   combined[idx] = {
      "faiss_score" : 0.0,
      "bm25_score":0.0
    }

   combined[idx]["bm25_score"] = (bm25_normalized[position])

 results = []

 for idx,scores in combined.items():
  hybrid_score=(0.5*scores["faiss_score"]+0.5*scores["bm25_score"])  
  results.append ({
    "index": idx,
    "faiss_score":scores["faiss_score"],
    "bm25_score":scores["bm25_score"],
    "hybrid_score":hybrid_score,
    "chunk":chunks[idx]
  })

 results.sort(key=lambda x: x["hybrid_score"],reverse=True)
 return results[:top_k]
 

def rerank_results(question,results,reranker,top_k=5):
 pairs =[[question,result["chunk"]] for result in results]
 scores=reranker.predict(pairs)
 for result,score in zip (results,scores):
  result["rerank_score"]=float(score)

 results.sort(key=lambda x: x["rerank_score"] , reverse=True)

 return results[:top_k]
 