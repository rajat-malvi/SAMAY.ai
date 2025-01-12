import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import google.generativeai as genai
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import List, Dict
import re
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key is missing. Ensure `GOOGLE_API_KEY` is set in the environment.")
    st.stop()

genai.configure(api_key=api_key)

FEEDBACK_FILE = "feedback.json"

def save_feedback_to_file(feedback_data):
    file_path = "feedback_data.json"
    
    # Check if the file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                # Try to load existing data from the file
                existing_data = json.load(f)
        except json.JSONDecodeError:
            # Handle the case where the JSON is invalid or the file is empty
            st.warning("The feedback data file is empty or corrupt. Initializing new file.")
            existing_data = []
    else:
        # If the file doesn't exist, initialize an empty list
        existing_data = []

    # Append the new feedback data
    existing_data.append(feedback_data)

    # Write the updated feedback data back to the file
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=4)


class EnhancedQASystem:
    def __init__(self, faiss_index_path: str = "faiss_index"):
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")

        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="Llama-3.1-70b-Versatile",
        )
        self.vector_store = FAISS.load_local(
            faiss_index_path, self.embeddings, allow_dangerous_deserialization=True
        )
        
        # Initialize additional components
        self.math_chain = LLMMathChain.from_llm(llm=self.llm)
        self.wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        self.search_tool = DuckDuckGoSearchRun()
        self.memory = ConversationBufferMemory()
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )

    def is_previous(self, text: str) -> bool:
        """
        Check if the query is requesting information about previous conversation.
        """
        previous_patterns = [
            r"\b(?:previous|above|earlier|last|before)\b.*?\b(?:question|answer|response|mention|said)\b",
            r"\b(?:what|as)\b.*?\b(?:mentioned|discussed|talked|said)\b.*?\b(?:before|earlier|above)\b",
            r"\btell\b.*?\b(?:more|about)\b.*?\b(?:that|this)\b",
            r"\brelated\b.*?\b(?:to|with)\b.*?\b(?:previous|above|earlier)\b",
            r"\bwhat\b.*?\b(?:you|we)\b.*?\b(?:mean|meant)\b.*?\b(?:by|with)\b",
            r"\bcan\b.*?\b(?:you|we)\b.*?\b(?:elaborate|explain)\b.*?\b(?:that|this)\b"
        ]
        
        return any(re.search(pattern, text.lower()) for pattern in previous_patterns)

    def contains_math(self, query: str) -> bool:
        math_patterns = r'[\d+\-*/^()=]+'
        return bool(re.search(math_patterns, query))
    
    def contains_greeting(self, text: str) -> bool:
        """Check if the input text contains a greeting."""
        greetings = r"\b(hi+|hello+|namaste|hey|good\s+(?:morning|afternoon|evening)|greetings|radhey|radhey-radhey|ram|ram-ram)\b"
        return bool(re.search(greetings, text.lower()))

    def get_greeting_response(self) -> dict:
        """Generate a direct greeting response using LLM."""
        prompt = """
            Provide a warm, friendly, and concise greeting response as a Gita Assistant. 
            Incorporate a relevant reference to the Bhagavad Gita or Yoga Sutras, and begin the response with 'Radhey-Radhey' or 'Jai Shri Ram' as part of the greeting. 
            Keep it under 2 sentences, welcoming and serene.
            """

        response = self.llm.predict(prompt)
        return {"response": response, "metadata": []}
    
    def contains_inappropriate_content(self, text):
        """
        Check if the text contains inappropriate or flagged content using regex patterns.
        """
        inappropriate_patterns = [
            r"\b(?:f\*?u\*?c\*?k|s\*?h\*?i\*?t|b\*?i\*?t\*?c\*?h)\b",  # Common profanities with optional masking
            r"\b(?:a\*?s\*?s|d\*?a\*?mn|c\*?u\*?n\*?t)\b",
            r"[^\w\s]{3,}",  # Strings with excessive symbols
            r"(.)\1{3,}",  # Repeated characters like "aaaa" or "!!!!"
        ]
        
        for pattern in inappropriate_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        if self.is_random_string(text):
            return True
        
        return False

    def is_random_string(self, text):
        """
        Check for random-like strings using length and entropy heuristics.
        """
        # Clean the text first
        cleaned_text = ' '.join(text.split())
        
        # Check for very short queries
        if len(cleaned_text.split()) < 5:
            # Allow common short queries like greetings
            if self.contains_greeting(cleaned_text):
                return False
            return True
        
        # Calculate entropy
        unique_chars = set(cleaned_text)
        entropy = len(unique_chars) / len(cleaned_text)
        return entropy > 0.8
    

    def get_inappropriate_content_response(self) -> dict:
        """
        Generate a standard response for inappropriate or random content.
        """
        return {
            "response": "I apologize, but your query appears to be irrelevant or inappropriate. Please ask a clear, relevant question about the Gita or related topics.",
            "metadata": []
        }


    def get_response(self, query: str, k: int = 3) -> dict:
        try:
            # Check for inappropriate content or random strings first
            if self.contains_inappropriate_content(query):
                return self.get_inappropriate_content_response()

            # Check for greetings
            if self.contains_greeting(query):
                return self.get_greeting_response()

            # For regular queries, proceed with full functionality
            response_parts = []
            
            # Check if query contains mathematical expressions
            if self.contains_math(query):
                math_response = self.math_chain.run(query)
                response_parts.append(f"Mathematical Result: {math_response}")

            # Get relevant documents from vector store
            docs = self.vector_store.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            metadata = [doc.metadata for doc in docs]

            # Add Wikipedia and web search results for non-greeting queries
            keyword = ''
            try:
                # Retrieve Wikipedia context
                wiki_result = self.wiki_tool.run(query) 
                if wiki_result:
                    # Extracting the first sentence or summary from the Wikipedia result
                    response_parts.append(f"Wikipedia Context: {wiki_result[:500]}...")
                    
                    # Extract the keyword for the Wikipedia link (e.g., if it's a proper noun or specific term)
                    # This approach ensures we safely extract the term from the wiki result
                    # Here, we assume that we want to link to the Wikipedia page for the query
                    keyword = query.strip().replace(" ", "_")  # Replace spaces with underscores for URL compatibility        
            except Exception as e:
                st.warning(f"Wikipedia retrieval error: {str(e)}")

            
            # Web search    
            try:
                search_result = self.search_tool.run(query)
                if search_result:
                    response_parts.append(f"Web Search Context: {search_result[:500]}...")
            except Exception as e:
                st.warning(f"Web search error: {str(e)}")

            # Get conversation history
            conversation_history = self.memory.load_memory_variables({})
            history_context = conversation_history.get("history", "")

            # Combine all context
            full_context = "\n\n".join(response_parts) + "\n\n" + context

            # Create enhanced prompt for concise responses
            if self.is_previous(query):
                prompt_template = """
                Based on the provided context, please answer the following question.

                Previous Conversation:
                {history}

                Current Context: {context}
                Current Question: {question}

                Provide a comprehensive response that includes:
                1. Direct answer to the question
                2. Relevant information from the source data
                3. Any additional context or explanations

                Answer:
                """
                
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["history", "context", "question"]
                )

                final_response = self.llm.predict(
                    prompt.format(
                        history=history_context,
                        context=full_context,
                        question=query
                    )
                )
            else:
                prompt_template = """
                Based on the following information, provide a short and precise response:

                Context: {context}
                Question: {question}

                Guidelines:
                1. Be direct and concise in your answer.
                2. Focus on the most relevant information.
                3. Use clear, simple language.
                4. Keep the response under 3-4 sentences unless more detail is explicitly requested.
                5. Always cite the source of the information. Indicate whether it comes from:
                - Retrieval Database (e.g., database name or ID)
                - Wiki search (include the URL or article title)
                - Books (include the chapter and verse number, if applicable)
                - Web search (include the URL or website name)
                6. When relevant, prioritize information from the Gita or other authoritative texts, and cite the chapter and verse explicitly.

                Answer:
                """

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )

                final_response = self.llm.predict(
                    prompt.format(
                        context=full_context,
                        question=query
                    )
                )




            # Update conversation memory
            self.memory.save_context(
                {"input": query},
                {"output": final_response}
            )
            
            #  wiki add
            for doc in metadata:
                if keyword:
                    doc['Wiki_search'] = f"https://en.wikipedia.org/wiki/{keyword}"


            return {"response": final_response, "metadata": metadata}
                
        except Exception as e:
            return {"response": f"Error generating response: {str(e)}", "metadata": []}
def extract_questions(metadata: List[dict]) -> List[str]:
    return [meta.get("question", "") for meta in metadata]


def main():
    st.set_page_config(page_title="🙏 Gita", layout="centered")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "feedback_history" not in st.session_state:
        st.session_state.feedback_history = []
    if "processed_feedback" not in st.session_state:
        st.session_state.processed_feedback = set()
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "is_submitting" not in st.session_state:
        st.session_state.is_submitting = False
    if "show_feedback_modal" not in st.session_state:
        st.session_state.show_feedback_modal = False
    if "app_feedback" not in st.session_state:
        st.session_state.app_feedback = {"rating": None, "comments": ""}

    # UI Layout
    st.title("🙏 Namaste: Your Gita Assistant")
    st.markdown(
        """
        **Ask any question about Gita**  
        🤖 Enhanced with math processing, Wikipedia integration, and conversation memory.
        """
    )

    # Feedback Button
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("Feedback"):
            st.session_state.show_feedback_modal = True

    # Show Feedback Modal
    if st.session_state.show_feedback_modal:
        st.markdown("---")
        st.markdown("### Submit Feedback 🙏")

        # Star Rating
        rating = st.radio(
            "How would you rate your experience?",
            options=[1, 2, 3, 4, 5],
            index=(st.session_state.app_feedback["rating"] - 1 if st.session_state.app_feedback["rating"] else 2),
            horizontal=True,
        )

        # Feedback Text Box
        feedback_text = st.text_area(
            "Additional Comments:",
            value=st.session_state.app_feedback["comments"],
            placeholder="Let us know how we can improve!",
        )

        # Submit Feedback Button
        if st.button("Submit Feedback", key="submit_feedback"):
            st.session_state.app_feedback["rating"] = rating
            st.session_state.app_feedback["comments"] = feedback_text

            # Prepare feedback dictionary
            feedback_data = {
                "type": "review",
                "rating": rating,
                "comments": feedback_text,
            }

            # Save to JSON file
            try:
                save_feedback_to_file(feedback_data)
                st.success("Thank you for your feedback! It has been saved.")
            except Exception as e:
                st.error(f"An error occurred while saving feedback: {e}")

            st.session_state.show_feedback_modal = False

        # Cancel Feedback Button
        if st.button("Cancel", key="cancel_feedback"):
            st.session_state.show_feedback_modal = False

        st.markdown("---")

    # Sidebar: Suggested Questions
    with st.sidebar:
        st.header("Suggested Questions")
        if st.session_state.chat_history:
            latest_chat = st.session_state.chat_history[-1]
            metadata_questions = extract_questions(latest_chat.get("metadata", {}))
            if metadata_questions:
                for idx, question in enumerate(metadata_questions):
                    if st.button(f"{idx + 1}. {question}", key=f"suggest_{idx}"):
                        st.session_state.user_query = question
            else:
                st.write("No questions available.")
        else:
            default_questions = [
                "How should we handle our desires?",
                "What is Karma?",
                "How to be Happy?",
            ]
            for idx, question in enumerate(default_questions):
                if st.button(f"{idx + 1}. {question}", key=f"default_{idx}"):
                    st.session_state.user_query = question

    # Chat History Display
    st.write("### Chat History")
    for idx, chat in enumerate(st.session_state.chat_history):
        st.write(f"**Q{idx + 1}:** {chat['query']}")
        st.write(f"**A{idx + 1}:** {chat['response']}")
        with st.expander(f"Metadata for Q{idx + 1}"):
            st.json(chat["metadata"])

        # Feedback Section
        feedback_key = f"{idx}_{chat['query']}"
        if feedback_key not in st.session_state.processed_feedback:
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"👍 Like", key=f"like_{feedback_key}"):
                    feedback_data = {
                        "type": "chat_feedback",
                        "query": chat["query"],
                        "feedback": "like",
                    }
                    save_feedback_to_file(feedback_data)
                    st.session_state.feedback_history.append(feedback_data)
                    st.session_state.processed_feedback.add(feedback_key)
            with col2:
                if st.button(f"👎 Dislike", key=f"dislike_{feedback_key}"):
                    feedback_data = {
                        "type": "chat_feedback",
                        "query": chat["query"],
                        "feedback": "dislike",
                    }
                    save_feedback_to_file(feedback_data)
                    st.session_state.feedback_history.append(feedback_data)
                    st.session_state.processed_feedback.add(feedback_key)
        else:
            feedback = next(
                (f for f in st.session_state.feedback_history if f["query"] == chat["query"]), None
            )
            if feedback:
                st.write(f"*Feedback recorded: {'👍' if feedback['feedback'] == 'like' else '👎'}*")

    # Input Query Section
    st.write("### Enter Your Query Below:")
    query = st.text_input("Enter your question:", st.session_state.user_query, key="user_query_input")

    if st.button("Submit", disabled=st.session_state.is_submitting):
        if query and (not st.session_state.chat_history or st.session_state.chat_history[-1]["query"] != query):
            st.session_state.is_submitting = True
            with st.spinner("Generating response..."):
                try:
                    qa_system = EnhancedQASystem()  # Replace with your logic
                    result = qa_system.get_response(query)
                    st.session_state.chat_history.append({
                        "query": query,
                        "response": result["response"],
                        "metadata": result["metadata"],
                    })
                    st.session_state.user_query = ""
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    st.session_state.is_submitting = False

    # Feedback Summary
    if st.session_state.feedback_history:
        with st.expander("### Feedback Summary", expanded=False):
            for feedback in st.session_state.feedback_history:
                feedback_icon = "👍" if feedback["feedback"] == "like" else "👎"
                st.write(f"**Query:** {feedback['query']} - Feedback: {feedback_icon}")


if __name__ == "__main__":
    main()