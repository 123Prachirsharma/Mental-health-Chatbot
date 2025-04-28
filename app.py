import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI(title="Mental Health Chatbot")

# Request model for chat input
class ChatRequest(BaseModel):
    message: str

# Initialize the LLM
def initialize_llm():
    llm = ChatGroq(
        temperature=0,
        groq_api_key="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",  # Replace with os.getenv("GROQ_API_KEY") in production
        model_name="llama3-70b-8192"
    )
    return llm

# Create or load the vector database
def create_vector_db():
    data_dir = "./data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError("No PDF documents found in ./data/. Please add PDFs.")
    
    print(f"[INFO] Loaded {len(documents)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"[INFO] Split into {len(texts)} text chunks.")
    
    if not texts:
        raise ValueError("No text chunks created. Ensure PDFs contain extractable text.")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    vector_db.persist()
    print("[INFO] ChromaDB created and data saved.")
    return vector_db

# Set up the QA chain
def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_template = """You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
{context}
user: {question}
chatbot:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Initialize chatbot components
print("Initializing Chatbot...")
llm = initialize_llm()
db_path = "./chroma_db"
if not os.path.exists(db_path):
    vector_db = create_vector_db()
else:
    print("[INFO] Loading existing vector database.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)

# HTML for the frontend
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f4f4f9; }
        h1 { text-align: center; color: #333; }
        .chat-container { max-width: 600px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .chat-box { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
        .message { margin: 5px 0; }
        .user { color: #007bff; }
        .bot { color: #28a745; }
        input { width: 100%; padding: 10px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
    </style>
</head>
<body>
    <h1>🧠 Compassionate Mental Health Chatbot</h1>
    <div class="chat-container">
        <div class="chat-box" id="chat-box"></div>
        <input type="text" id="user-input" placeholder="Type your message here..." />
        <button onclick="sendMessage()">Send</button>
    </div>
    <script>
        async function sendMessage() {
            const input = document.getElementById('user-input');
            const chatBox = document.getElementById('chat-box');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Display user message
            chatBox.innerHTML += `<div class="message user">You: ${message}</div>`;
            input.value = '';
            chatBox.scrollTop = chatBox.scrollHeight;
            
            // Send message to backend
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                const data = await response.json();
                
                // Display bot response
                chatBox.innerHTML += `<div class="message bot">Chatbot: ${data.response}</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            } catch (error) {
                chatBox.innerHTML += `<div class="message bot">Chatbot: Error occurred. Please try again.</div>`;
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        }
        
        // Allow sending message with Enter key
        document.getElementById('user-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
"""

# Serve the frontend
@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTML_CONTENT

# Chat endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    response = qa_chain.invoke({"query": request.message})
    return {"response": response["result"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)