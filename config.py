import requests
import streamlit as st

API_KEY = st.secrets["GEMINI_API_KEY"]

MODEL_NAME="gemini-3.5-flash-lite"

URL = (
         "https://generativelanguage.googleapis.com/"
         "v1beta/models/"	
         + MODEL_NAME
         + ":generateContent"
)



def generate_sql_with_gemini(prompt):

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1000
        }
    }

    try:
      response = requests.post(
        URL,
        headers=headers,
        json=data,
        timeout=60
     )

      response.raise_for_status()
      result = response.json()
      candidates = result.get("candidates", [])
      if not candidates:
            raise Exception(
                "Gemini returned no candidates."
            )

      parts = candidates[0].get(
            "content", {}
          ).get("parts", [])

      answer = "".join(
            part.get("text", "")
            for part in parts
        ).strip()

      if not answer:
            raise Exception(
                "Gemini returned an empty response."
            )

      return answer

    except requests.exceptions.Timeout:
        raise Exception(
            "Gemini API request timed out."
        )

    except requests.exceptions.RequestException as error:
        raise Exception(
            f"Gemini API connection failed: {error}"
        )

