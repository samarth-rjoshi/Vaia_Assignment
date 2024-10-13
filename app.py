import streamlit as st
import requests

# Define FastAPI base URL
BASE_URL = "http://127.0.0.1:8000"

# Streamlit App Title
st.title("URL Text Extractor and Query System")

# Option to parse a URL and extract content
st.header("Step 1: Extract Text from a URL")
url = st.text_input("Enter the URL to parse", placeholder="https://example.com")

if st.button("Extract Text"):
    if url:
        with st.spinner("Extracting text from the URL..."):
            response = requests.post(f"{BASE_URL}/url-parser", json={"url": url})
            if response.status_code == 200:
                st.success("Text successfully extracted and stored!")
                st.write(response.json())
            else:
                st.error(f"Error: {response.json()['detail']}")
    else:
        st.warning("Please enter a URL.")

# Option to query the extracted content
st.header("Step 2: Query the Extracted Text")
query = st.text_input("Enter your query", placeholder="What is this article about?")

if st.button("Get Answer"):
    if query:
        with st.spinner("Fetching the most relevant information..."):
            response = requests.post(f"{BASE_URL}/query", json={"query": query})
            if response.status_code == 200:
                st.success("Response received!")
                st.write("Answer:", response.json().get("llm_response", "No response generated"))
            else:
                st.error(f"Error: {response.json()['detail']}")
    else:
        st.warning("Please enter a query.")
