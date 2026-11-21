
import streamlit as st

st.set_page_config(
    page_title="AI Document Assistant",
    layout="wide"
)

st.title("AI Document Assistant")

st.success("Streamlit application is running!")

st.write("Deployment test successful.")

name = st.text_input("Enter your name")

if name:
    st.write(f"Welcome, {name}!")
