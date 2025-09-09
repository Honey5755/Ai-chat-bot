"""Core chatbot logic using LangChain, ChromaDB (RAG), and SQLite for simple persistent memory."""
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from pathlib import Path
import sqlite3, json, uuid

CHROMA_DIR = Path(__file__).parent / '.chromadb'
DB_FILE = Path(__file__).parent / 'db' / 'conversations.sqlite3'
DB_FILE.parent.mkdir(exist_ok=True)

def get_vector_store():
    client_settings = {"chroma_db_impl":"duckdb+parquet", "persist_directory": str(CHROMA_DIR)}
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY'))
    vectordb = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings, collection_name='support_collection')
    return vectordb

def init_sqlite():
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS conversations (session_id TEXT PRIMARY KEY, data TEXT)''')
    conn.commit()
    conn.close()

def save_conversation(session_id, memory_dict):
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute('REPLACE INTO conversations (session_id, data) VALUES (?, ?)', (session_id, json.dumps(memory_dict)))
    conn.commit()
    conn.close()

def load_conversation(session_id):
    conn = sqlite3.connect(str(DB_FILE))
    c = conn.cursor()
    c.execute('SELECT data FROM conversations WHERE session_id=?', (session_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

class SupportChatbot:
    def __init__(self):
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('Please set OPENAI_API_KEY before using the chatbot.')
        init_sqlite()
        self.vectordb = get_vector_store()
        self.llm = ChatOpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo', temperature=0.0)
        self.chain = ConversationalRetrievalChain.from_llm(self.llm, self.vectordb.as_retriever(search_kwargs={"k":3}), return_source_documents=True)
        self.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    def ask(self, session_id, question):
        # load session memory from sqlite (if any)
        mem = load_conversation(session_id) or {}
        # we only persist a small "chat_history" list
        chat_history = mem.get('chat_history', [])
        # call chain with chat history
        result = self.chain({"question": question, "chat_history": chat_history})
        # update local memory and save
        chat_history.append(("user", question))
        chat_history.append(("assistant", result.get('answer')))
        mem['chat_history'] = chat_history[-50:]  # keep last 50 turns
        save_conversation(session_id, mem)
        # return result with sources
        sources = [doc.metadata for doc in result.get('source_documents', [])]
        return result.get('answer'), sources
