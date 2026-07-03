import streamlit as st
import os
import sys
import tempfile
from dotenv import load_dotenv

# Ensure the 'code' directory and 'code/src' is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.rag_engine import RAGEngine

# Set Page Config
st.set_page_config(
    page_title="RetrievalGPT - Premium AI RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling - Dark Premium AI Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main Layout Styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #0B1020 !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(11, 16, 32, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] {
        background-color: #121826 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    .main-title {
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.75rem;
        font-weight: 800;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .sub-title {
        font-size: 1rem;
        color: #94A3B8;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Card Container Styling */
    .card {
        background-color: #111827;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Citation Panel */
    .citation-card {
        background-color: #121826;
        border-left: 4px solid #6366F1;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.88rem;
        color: #94A3B8;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .citation-card:hover {
        transform: scale(1.008);
        border-color: rgba(99, 102, 241, 0.3);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.05);
    }
    
    /* Interactive Streamlit Elements Override */
    div.stButton > button {
        background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
        color: #F8FAFC !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 0.5rem 1.25rem !important;
        font-weight: 600 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100%;
        font-size: 0.9rem !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
        border-color: rgba(255, 255, 255, 0.18) !important;
    }
    
    /* File Uploader override */
    [data-testid="stFileUploader"] {
        background-color: #111827 !important;
        border: 1px dashed rgba(99, 102, 241, 0.3) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #8B5CF6 !important;
        background-color: rgba(99, 102, 241, 0.02) !important;
    }
    
    /* Sidebar Input widgets */
    div[data-baseweb="input"] {
        background-color: #121826 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
    }
    
    [data-testid="stExpander"] {
        background-color: #111827 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Chat Experience Custom Styles */
    .chat-bubble-user {
        background-color: #6366F1;
        color: #F8FAFC;
        padding: 0.75rem 1.25rem;
        border-radius: 16px 16px 4px 16px;
        margin-bottom: 1.25rem;
        display: inline-block;
        max-width: 80%;
        float: right;
        clear: both;
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .chat-bubble-assistant {
        background-color: #111827;
        color: #F8FAFC;
        padding: 0.75rem 1.25rem;
        border-radius: 16px 16px 16px 4px;
        margin-bottom: 1.25rem;
        display: inline-block;
        max-width: 80%;
        float: left;
        clear: both;
        border: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.95rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .think-block {
        background-color: rgba(99, 102, 241, 0.03) !important;
        border-left: 3px solid #8B5CF6 !important;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        font-style: italic;
        color: #C084FC !important;
        font-size: 0.9rem;
    }
    
    /* Provider status tag */
    .provider-tag {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .tag-ollama { background-color: rgba(248, 250, 252, 0.08); color: #F8FAFC; border: 1px solid rgba(248, 250, 252, 0.15); }
    .tag-gemini { background-color: rgba(59, 130, 246, 0.12); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.25); }
    .tag-groq { background-color: rgba(245, 158, 11, 0.12); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.25); }
    .tag-hf { background-color: rgba(139, 92, 246, 0.12); color: #A78BFA; border: 1px solid rgba(139, 92, 246, 0.25); }
    .tag-mock { background-color: rgba(239, 68, 68, 0.12); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.25); }
    
    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0B1020;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Header Section
st.markdown("<div class='main-title'>RetrievalGPT</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Local-first conversational reasoning engine powered by open-source fallbacks</div>", unsafe_allow_html=True)

# Sidebar Config Panel
with st.sidebar:
    st.markdown("### ⚡ AI Workspace")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
        
    st.markdown("---")

    # Collapsible Settings Panels
    with st.expander("🔑 Cloud Provider Keys", expanded=True):
        google_api_key = st.text_input(
            "Gemini API Key (Google)",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Free API Key from Google AI Studio"
        )
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Free API Key from Groq Console"
        )
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

        hf_token = st.text_input(
            "Hugging Face Token",
            type="password",
            value=os.getenv("HUGGINGFACEHUB_API_TOKEN", ""),
            help="Free API Access Token from Hugging Face Settings"
        )
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    with st.expander("💻 Local Ollama Config", expanded=False):
        ollama_url = st.text_input(
            "Ollama Endpoint",
            value=os.getenv("OLLAMA_API_URL", "http://localhost:11434"),
            help="Your local Ollama service URL."
        )
        if ollama_url:
            os.environ["OLLAMA_API_URL"] = ollama_url

        ollama_model = st.text_input(
            "Ollama Model",
            value=os.getenv("OLLAMA_MODEL", "llama3.1"),
            help="Local model name (e.g. llama3.1, phi3, gemma3, mistral)."
        )
        if ollama_model:
            os.environ["OLLAMA_MODEL"] = ollama_model

    with st.expander("⚙️ Tuning Parameters", expanded=False):
        chunk_size = st.slider("Chunk Size (Words)", min_value=100, max_value=1000, value=400, step=50)
        chunk_overlap = st.slider("Chunk Overlap (Words)", min_value=10, max_value=300, value=80, step=10)
        top_k = st.slider("Documents to Retrieve", min_value=1, max_value=10, value=4)
        
    st.markdown("---")
    
    # Document Upload Section
    st.markdown("### 📁 Documents")
    uploaded_files = st.file_uploader(
        "Ingest PDF / TXT / MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Upload text resources to index them into your local vectors."
    )
    
    # Process files
    if uploaded_files:
        new_files = [f.name for f in uploaded_files]
        if new_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = new_files
            with st.spinner("Ingesting documents..."):
                st.session_state.rag_engine.documents = []
                st.session_state.rag_engine.chunks = []
                
                for uploaded_file in uploaded_files:
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                        temp_file.write(uploaded_file.read())
                        temp_path = temp_file.name
                    
                    try:
                        doc_id = st.session_state.rag_engine.load_document(temp_path)
                        st.session_state.rag_engine.documents[doc_id]["source"] = uploaded_file.name
                        num_chunks = st.session_state.rag_engine.index_document(doc_id, chunk_size, chunk_overlap)
                        st.sidebar.success(f"Indexed {uploaded_file.name} ({num_chunks} chunks)")
                    except Exception as e:
                        st.sidebar.error(f"Error loading {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(temp_path)

    st.markdown("---")
    
    # Mock Conversation History Panel
    st.markdown("### 💬 Chat History")
    conversations = [
        "📄 RAG System Architecture Review",
        "🔬 Local LLM Performance Test",
        "📚 Document Ingestion Overview"
    ]
    for conv in conversations:
        st.markdown(f"""
        <div style='padding: 0.5rem 0.75rem; border-radius: 8px; background-color: #111827; margin-bottom: 0.5rem; border: 1px solid rgba(255,255,255,0.05); font-size: 0.85rem; color: #94A3B8;'>
            {conv}
        </div>
        """, unsafe_allow_html=True)
        
    # Stats status
    total_docs = len(st.session_state.rag_engine.documents)
    total_chunks = len(st.session_state.rag_engine.chunks)
    
    st.markdown("---")
    st.markdown(f"📄 Docs: `{total_docs}` | 🧩 Chunks: `{total_chunks}`")
    st.markdown("🔮 Vectors: `all-MiniLM-L6-v2 (Local)`")

# Main Interface Options
col1, col2 = st.columns([4, 1])
with col1:
    enable_rewrite = st.toggle("Enable DeepRetrieval Reasoning (Query Rewriting)", value=True)
with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Chat Area Container
chat_container = st.container()

with chat_container:
    # Display existing chat messages
    for speaker, text, thought, sources in st.session_state.chat_history:
        if speaker == "User":
            st.markdown(f"<div class='chat-bubble-user'>{text}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🤖"):
                if thought:
                    with st.expander("💭 Thought Process (DeepRetrieval)", expanded=False):
                        st.markdown(f"<div class='think-block'>{thought}</div>", unsafe_allow_html=True)
                st.markdown(text)
                if sources:
                    with st.expander("🔍 Citations & Sources", expanded=False):
                        for idx, (source, text_preview, score) in enumerate(sources):
                            st.markdown(f"""
                            <div class='citation-card'>
                                <strong>Source {idx+1}: {source}</strong> (Similarity Score: {score:.2f})<br/>
                                <em>"{text_preview}"</em>
                            </div>
                            """, unsafe_allow_html=True)

# User Query input
if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    st.markdown(f"<div class='chat-bubble-user'>{prompt}</div>", unsafe_allow_html=True)
    
    thought_process = ""
    rewritten_query = prompt
    retrieved_chunks = []
    
    # 1. Query Rewriting Phase
    if enable_rewrite:
        with st.status("🧠 DeepRetrieval: Formulating query terms...") as status:
            thought_process, rewritten_query = st.session_state.rag_engine.rewrite_query(
                query=prompt,
                ollama_url=ollama_url,
                ollama_model=ollama_model
            )
            st.markdown(f"**Thought:** {thought_process}")
            st.markdown(f"**Reformulated Search Terms:** `{rewritten_query}`")
            status.update(label="Query terms constructed successfully!", state="complete")
            
    # 2. Retrieval Phase
    with st.status("🔍 Searching local vector database...") as status:
        retrieved_chunks = st.session_state.rag_engine.retrieve(rewritten_query, top_k=top_k)
        status.update(label=f"Retrieved {len(retrieved_chunks)} relevant sources.", state="complete")
        
    # 3. Response Generation Phase
    with st.spinner("🤖 Formulating final response..."):
        sources_list = []
        for chunk, score in retrieved_chunks:
            sources_list.append((chunk["source"], chunk["text"][:250], score))
            
        history_tuples = [(role, text) for role, text, _, _ in st.session_state.chat_history]
        
        response = st.session_state.rag_engine.generate_answer(
            query=prompt,
            context_chunks=retrieved_chunks,
            chat_history=history_tuples,
            ollama_url=ollama_url,
            ollama_model=ollama_model
        )
        
        # Display assistant response in chat
        with st.chat_message("assistant", avatar="🤖"):
            if thought_process:
                with st.expander("💭 Thought Process (DeepRetrieval)", expanded=True):
                    st.markdown(f"<div class='think-block'>{thought_process}</div>", unsafe_allow_html=True)
            st.markdown(response)
            
            if sources_list:
                with st.expander("🔍 Citations & Sources", expanded=False):
                    for idx, (source, text_preview, score) in enumerate(sources_list):
                        st.markdown(f"""
                        <div class='citation-card'>
                            <strong>Source {idx+1}: {source}</strong> (Similarity Score: {score:.2f})<br/>
                            <em>"{text_preview}"</em>
                        </div>
                        """, unsafe_allow_html=True)
                        
        # Append to session state chat history
        st.session_state.chat_history.append((
            "User", prompt, "", []
        ))
        st.session_state.chat_history.append((
            "Assistant", response, thought_process, sources_list
        ))
        
        st.rerun()
