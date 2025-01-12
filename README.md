# 🙏 SAMAY.ai

**Spritual Assistance and Meditation Aid for You**

---

## Overview
The **SAMAY.ai** project is an AI-powered application designed to answer questions about the Bhagavad Gita and related topics. The assistant leverages advanced natural language processing (NLP), mathematical computation, and external knowledge retrieval for generating meaningful responses. Built using **Streamlit**, **LangChain**, and **Google Generative AI**, the application offers an interactive and insightful user experience.

---

## Features
- **Semantic Search**: Uses **FAISS** to perform similarity searches on embedded documents.
- **Natural Language Understanding**: Integrates **ChatGroq** for generating context-aware responses.
- **Mathematical Capabilities**: Processes mathematical expressions using `LLMMathChain`.
- **Knowledge Retrieval**: Retrieves contextual information from Wikipedia and web search tools.
- **Feedback System**: Allows users to submit ratings and comments for improving the system.
- **Streamlit-based UI**: Provides an intuitive interface for interaction and feedback.

---

## Requirements
- Python 3.8+
- Key Dependencies:
  - Streamlit
  - LangChain
  - FAISS
  - google-generative-ai
  - python-dotenv

---

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/rajat-malvi/SAMAY.ai.git>
   cd SAMAY.ai
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   - Create a `.env` file in the project directory with the following:
     ```env
     GOOGLE_API_KEY=your_google_api_key_here
     GROQ_API_KEY=your_groq_api_key_here
     ```

4. **Prepare the FAISS Index**:
   - Place the FAISS index in the specified path (`faiss_index`) or update the path in the code.

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## Usage
1. Launch the application using Streamlit.
2. Enter queries about the Bhagavad Gita in the input field.
3. Review the responses and explore suggestions in the sidebar.
4. Submit feedback to help improve the system.

---

## Contributing

- **Rajat Malviya**: [GitHub Profile](https://github.com/rajat-malvi)
- **Raushan Kumar**: [GitHub Profile](https://github.com/raushan22882917)
---

