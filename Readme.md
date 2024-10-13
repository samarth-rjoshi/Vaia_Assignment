# URL Text Extractor and Query System

This project is a web application built with FastAPI and Streamlit that allows users to extract text from a given URL and query the extracted content. It utilizes Langchain for managing document embeddings and a Hugging Face language model for generating responses based on user queries.

## Features

- **Text Extraction**: Extracts text from a specified URL, including text from linked pages up to a specified depth.
- **Embedding Storage**: Stores extracted text as embeddings in a vector database (Chroma).
- **Querying**: Allows users to query the extracted text and receive contextually relevant responses generated by a language model.

## Technologies Used

- **FastAPI**: For building the backend API.
- **Streamlit**: For creating the user interface.
- **Langchain**: For managing document embeddings and querying.
- **Hugging Face**: For text generation and embeddings.
- **BeautifulSoup**: For parsing HTML content from URLs.
- **ChromaDB**: For storing document embeddings.

## Getting Started

### Prerequisites

Make sure you have the following installed on your machine:

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

   
   git clone https://github.com/samarth-rjoshi/Vaia_Assugnment.git
   
   cd Vaia_Assugnment
   

3. Install the required dependencies:

   
   pip install -r requirements.txt
   

4. Set up your Hugging Face API token:

   Replace `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` in `main.py` with your own Hugging Face API token.

### Running the Application

1. Start the FastAPI backend:

   python main.py

2. In a new terminal, start the Streamlit frontend:

   streamlit run app.py


3. Open your web browser and go to `http://localhost:8501` to access the Streamlit app.

### Usage

1. **Step 1: Extract Text from a URL**
   - Enter a valid URL in the input field and click "Extract Text."
   - If successful, you will see a success message and information about the extracted content.

2. **Step 2: Query the Extracted Text**
   - Enter your query in the provided input field and click "Get Answer."
   - You will receive a generated answer based on the extracted content.


## License

This project is licensed under the MIT License. See the LICENSE file for details.
