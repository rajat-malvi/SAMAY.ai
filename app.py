from flask import Flask, request, jsonify
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
from datetime import datetime
from flask_cors import CORS
from UserSession import UserSession


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "PUT", "DELETE"], "allow_headers": "*"}})

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
grq_api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Google API key is missing. Ensure `GOOGLE_API_KEY` is set in the environment.")

genai.configure(api_key=api_key)


# Initialize feedback storage
FEEDBACK_FILE = "feedback_data.json"
# Dictionary to store user sessions
user_sessions = {}
def save_feedback_to_file(feedback_data):
    try:
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        
        feedback_data["timestamp"] = datetime.now().isoformat()
        existing_data.append(feedback_data)
        
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(existing_data, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return False

class EnhancedQASystem:
    def __init__(self, faiss_index_path: str = "faiss_index"):
        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")

        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGroq(
            groq_api_key=grq_api_key,
            model_name="llama-3.3-70b-versatile",
        )
        self.vector_store = FAISS.load_local(
            faiss_index_path, self.embeddings, allow_dangerous_deserialization=True
        )
        
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

        response = self.llm.invoke(prompt)

        return {"response": response.content, "metadata": []}
    
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
    

    def get_response(self, query: str, user_session: UserSession, k: int = 3) -> dict:
        try:

            if user_session.should_clear_history():
                user_session.clear_history()

            user_session.update_activity()

            
            # Check for inappropriate content or random strings first
            # if self.contains_inappropriate_content(query):
            #     return self.get_inappropriate_content_response()

            if self.contains_greeting(query):
                return self.get_greeting_response()

            response_parts = []
            docs = self.vector_store.similarity_search(query, k=k)
            context = "\n\n".join([doc.page_content for doc in docs])
            metadata = [doc.metadata for doc in docs]

            # Add Wikipedia and web search results
            keyword = ''
            try:
                query1 = f"{query} related to Bhagavad Gita and The Yoga Sutras of Patanjali"
                wiki_result = self.wiki_tool.run(query1)
                if wiki_result:
                    response_parts.append(f"Wikipedia Context: {wiki_result[:500]}...")
                    keyword = query.strip().replace(" ", "_")
            except Exception as e:
                print(f"Wikipedia retrieval error: {str(e)}")
            
            # try:
            #     search_result = self.search_tool.run(query)
            #     if search_result:
            #         response_parts.append(f"Web Search Context: {search_result[:500]}...")
            # except Exception as e:
            #     print(f"Web search error: {str(e)}")

            conversation_history = user_session.memory.load_memory_variables({})
            history_context = conversation_history.get("history", "")
            full_context = "\n\n".join(response_parts) + "\n\n" + context
            

            # Use appropriate prompt template based on query type
            if self.is_previous(query):
                prompt_template = """
                Based on the sacred teachings of the Bhagavad Gita and The Yoga Sutras of Patanjali, please answer the following question.
                Previous Conversation:
                {history}
                
                Current Context:
                {context}
                
                Current Question:
                {question}
                
                Provide a comprehensive response that includes:
                1. A direct answer based on the teachings
                2. Relevant spiritual or philosophical insights
                3. Additional context or explanations where appropriate
                
                Answer:
                """
                
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["history", "context", "question"]
                )

                final_response = self.llm.invoke(
                    prompt.format(
                        history=history_context,
                        context=full_context,
                        question=query
                    )
                )
            else:
                prompt_template = """
                Based on the following teachings from the Bhagavad Gita and The Yoga Sutras of Patanjali, provide a concise and accurate response:
                Context:
                {context}
                
                Question:
                {question}
                
                Guidelines:
                1. Provide a direct and precise answer
                2. Use clear and simple language
                3. Keep the response concise
                4. Cite sources where applicable
                5. Link to practical applications when relevant
                
                Answer:
                """

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )

                final_response = self.llm.invoke(
                    prompt.format(
                        context=full_context,
                        question=query
                    )
                )
                
                print('fianll responce ', final_response)
            self.memory.save_context(
                {"input": query},
                {"output": final_response.content}
            )
            
            if keyword:
                for doc in metadata:
                    doc['Wiki_search'] = f"https://en.wikipedia.org/wiki/{keyword}"

            return {"response": final_response.content, "metadata": metadata}
                
        except Exception as e:
            return {"response": f"Error generating response: {str(e)}", "metadata": []}

# Initialize QA System
qa_system = None
try:
    qa_system = EnhancedQASystem()
except Exception as e:
    print(f"Error initializing QA System: {str(e)}")

def clean_inactive_sessions():
    """Remove inactive sessions older than 30 minutes"""
    current_time = datetime.now()
    inactive_users = [
        user_id for user_id, session in user_sessions.items()
        if (current_time - session.last_activity).total_seconds() / 60 > 30
    ]
    for user_id in inactive_users:
        del user_sessions[user_id]

@app.route('/chat', methods=['POST','GET'])
def chat():
    try:
        data = request.get_json()
        print(data)
        
        if not data or 'query' not in data:
            return jsonify({"error": "Missing query in request body"}), 400
        
        # Get or create user session using user_id or session_id
        user_id = data.get('user_id', request.remote_addr)  # Fallback to IP address if no user_id
        
        if user_id not in user_sessions:
            user_sessions[user_id] = UserSession(user_id)
        
        
        query = data['query'].strip()
        
        if not query:
            return jsonify({"error": "Empty query"}), 400
            
        if not qa_system:
            return jsonify({"error": "QA System not initialized"}), 500
            
        result = qa_system.get_response(query, user_sessions[user_id])
        
        return jsonify({
            "query": query,
            "response": result["response"],
            "metadata": result["metadata"]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Missing feedback data"}), 400
            
        # Validate feedback data
        required_fields = {
            'chat_feedback': ['type', 'query', 'feedback'],
            'review': ['type', 'rating', 'comments']
        }
        
        feedback_type = data.get('type')
        if not feedback_type or feedback_type not in required_fields:
            return jsonify({"error": "Invalid feedback type"}), 400
            
        for field in required_fields[feedback_type]:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        if save_feedback_to_file(data):
            return jsonify({"message": "Feedback saved successfully"})
        else:
            return jsonify({"error": "Error saving feedback"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "qa_system": "initialized" if qa_system else "not initialized"})

if __name__ == "__main__":
    app.run(debug=True)
