import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
import os
from typing import List, Dict
from dotenv import load_dotenv

class DataProcessor:
    def __init__(self, api_key: str = None):
        """
        Initialize the DataProcessor with Google API credentials
        
        Args:
            api_key (str): Google API key. If None, will try to load from environment
        """
        # Load environment variables if no API key provided
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("GOOGLE_API_KEY")
            
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing")
            
        genai.configure(api_key=api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
    def process_csv_files(self, file_paths: List[str], output_dir: str = "faiss_index") -> bool:
        """
        Process multiple CSV files and create a FAISS vector store
        
        Args:
            file_paths (List[str]): List of paths to CSV files
            output_dir (str): Directory to save the FAISS index
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            all_documents = []
            
            # Process each CSV file
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                    
                print(f"Processing file: {file_path}")
                df = pd.read_csv(file_path)
                
                # Create documents from each row
                for _, row in df.iterrows():
                    content = ""
                    metadata = {'source_file': file_path}
                    
                    # Convert each column into content and metadata
                    for column in df.columns:
                        content += f"{column}: {row[column]}\n"
                        metadata[column] = str(row[column])
                    
                    all_documents.append({
                        'content': content,
                        'metadata': metadata
                    })
            
            # Create vectors from documents
            texts = [doc['content'] for doc in all_documents]
            metadatas = [doc['metadata'] for doc in all_documents]
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create and save FAISS index
            vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            vector_store.save_local(output_dir)
            
            print(f"Successfully processed {len(file_paths)} files and saved FAISS index to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Error processing CSV files: {str(e)}")
            return False

def main():
    # Example usage
    csv_paths = [
        "./Bhagwad_Gita_Verses_English_final.csv",
        "./Patanjali_Yoga_Sutras_Verses_English_Questions.csv"
    ]
    
    try:
        processor = DataProcessor()
        success = processor.process_csv_files(csv_paths)
        
        if success:
            print("CSV processing completed successfully")
        else:
            print("Failed to process CSV files")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main()