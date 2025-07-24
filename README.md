# CorpDocs: Corporate Documentation Chatbot with RAG

CorpDocs is an AI-powered assistant for managing, generating, and verifying corporate documentation. It leverages Retrieval-Augmented Generation (RAG) using LangChain, LangGraph, and Streamlit to provide chat-based access to your corporate documents, with compliance checks and source citations.

## Features

- **Chat Interface:** Ask questions about your corporate documents and get detailed, cited answers.
- **Document Upload:** Supports PDF, TXT, EPUB, DOC, and DOCX files.
- **Compliance Checking:** Automatically reviews generated documentation for adherence to corporate standards.
- **Human Feedback Integration:** Incorporates corrective feedback when compliance issues are detected.
- **Source Citations:** Retrieves and attaches source citations to the final document.


## Getting Started

### 1. Clone the Repository

```sh
git clone https://github.com/yourusername/corporate-doc-chatbot-rag.git
cd corporate-doc-chatbot-rag
```

```
pip install -r requirements.txt
```

```
streamlit run streamlit_app.py