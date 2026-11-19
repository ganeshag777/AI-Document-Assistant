import config as gt

def generate_answer (user_question,context): 
 prompt =f"""

You are answering questions about an uploaded document.

Important instructions:

1. Read all the context carefully.

2. Answer using the context.

3. Use your own words when explaining.

4. Do not require an exact matching sentence.

5. If the context gives partial information, explain the
   available information.

6. Only say "I could not find the answer in the document"
   when the context is genuinely unrelated to the question.
7.Do not invent information.



Document_Context:
"""
 prompt += str(context) +"""

User Question:
"""
 prompt += user_question 
 answer =gt.generate_sql_with_gemini (prompt)
 return answer 



