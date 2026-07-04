import streamlit as st
import os
import sys
import tempfile
import time
from dotenv import load_dotenv

# Ensure the 'code' directory and 'code/src' is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.rag_engine import RAGEngine
from src.core.observability import telemetry
from src.core.citation import CitationEngine
from src.core.memory import ConversationalMemory

# Set Page Config
st.set_page_config(
    page_title="RetrievalGPT Enterprise - Premium AI Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Glassmorphic Layout and Typography
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Glassmorphic Theme background */
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 50%, #0d122b 0%, #060814 100%) !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="stHeader"] {
        background-color: rgba(6, 8, 20, 0.5) !important;
        backdrop-filter: blur(20px) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 14, 35, 0.8) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Titles and typography */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .gradient-title {
        background: linear-gradient(135deg, #a5b4fc, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic Panel Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin-bottom: 1.5rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    }
    
    .metric-title {
        font-size: 0.8rem;
        color: #818cf8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    /* Interactive Citations sidecards */
    .cite-side-card {
        background: rgba(99, 102, 241, 0.05) !important;
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.75rem !important;
        transition: all 0.2s ease;
    }
    .cite-side-card:hover {
        background: rgba(99, 102, 241, 0.1) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
        transform: translateY(-1px);
    }
    
    /* Chat Bubble Overrides */
    .user-bubble {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
        color: #FFFFFF !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 0.8rem 1.2rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.2) !important;
        display: inline-block;
        max-width: 85%;
        float: right;
        clear: both;
    }
    
    .assistant-bubble {
        background: rgba(255, 255, 255, 0.04) !important;
        color: #E2E8F0 !important;
        border-radius: 16px 16px 16px 4px !important;
        padding: 1rem 1.4rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15) !important;
        display: inline-block;
        max-width: 85%;
        float: left;
        clear: both;
    }

    .think-bubble {
        background: rgba(139, 92, 246, 0.05) !important;
        border-left: 3px solid #8b5cf6 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 4px !important;
        color: #d8b4fe !important;
        font-size: 0.88rem !important;
        font-style: italic !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Input adjustments */
    div[data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
    }
    
    /* Slider/Toggle tweaks */
    div.stButton > button {
        background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: #fff !important;
        padding: 0.5rem 1rem !important;
    }
    
    .chat-history-item {
        padding: 0.6rem 0.8rem;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
        cursor: pointer;
    }
    .chat-history-item:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session state variables
if "rag_engine" not in st.session_state:
    with st.spinner("Initializing Enterprise Hybrid RAG Engine..."):
        st.session_state.rag_engine = RAGEngine()
if "active_session" not in st.session_state:
    st.session_state.active_session = "Default Session"
if "memory" not in st.session_state:
    st.session_state.memory = ConversationalMemory(
        session_id=st.session_state.active_session,
        call_llm_fn=st.session_state.rag_engine._call_free_llm
    )
if "conversations" not in st.session_state:
    st.session_state.conversations = ["Default Session", "⚡ Systems Architecture Chat", "📂 Financial Reports Review"]
if "citations_list" not in st.session_state:
    st.session_state.citations_list = []

# Header Banner
st.markdown("<div class='gradient-title'>RetrievalGPT Enterprise</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Production-grade Agentic Hybrid RAG Platform with Structured Telemetry & Guardrails</div>", unsafe_allow_html=True)

# Sidebar layout
with st.sidebar:
    st.markdown("### ⚙️ Workspace Configuration")
    
    # Session Management
    selected_session = st.selectbox(
        "Select Active Session",
        options=st.session_state.conversations,
        index=st.session_state.conversations.index(st.session_state.active_session)
    )
    if selected_session != st.session_state.active_session:
        st.session_state.active_session = selected_session
        st.session_state.memory = ConversationalMemory(
            session_id=selected_session,
            call_llm_fn=st.session_state.rag_engine._call_free_llm
        )
        st.rerun()
        
    if st.button("➕ New Conversation Session", use_container_width=True):
        new_name = f"Session {len(st.session_state.conversations) + 1}"
        st.session_state.conversations.append(new_name)
        st.session_state.active_session = new_name
        st.session_state.memory = ConversationalMemory(
            session_id=new_name,
            call_llm_fn=st.session_state.rag_engine._call_free_llm
        )
        st.rerun()

    st.markdown("---")

    # API Keys Expander
    with st.expander("🔑 Access Keys & Tokens", expanded=True):
        google_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", "")
        )
        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key
            
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", "")
        )
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

        tavily_key = st.text_input(
            "Tavily Search API Key (Web Fallback)",
            type="password",
            value=os.getenv("TAVILY_API_KEY", "")
        )
        if tavily_key:
            os.environ["TAVILY_API_KEY"] = tavily_key
            
    # Tuning Panel
    with st.expander("🛠️ Advanced Tuning", expanded=False):
        strategy = st.selectbox("Search Strategy", ["Hybrid", "Dense-Only", "Sparse-Only"])
        chunk_size = st.slider("Chunk Size (Words)", 100, 1000, 400, 50)
        chunk_overlap = st.slider("Overlap (Words)", 10, 300, 80, 10)
        top_k = st.slider("Chunks to Retrieve", 1, 15, 5)
        web_search_fallback = st.toggle("Enable Web Search Fallback", value=True)

    # Metadata Filters
    with st.expander("🔍 Metadata Filters", expanded=False):
        doc_options = ["All"] + [d["source"] for d in st.session_state.rag_engine.documents]
        selected_doc = st.selectbox("Filter by Document", options=doc_options)
        selected_page = st.text_input("Filter by Page Number", value="", help="e.g. 5 or Leave blank")
        selected_chapter = st.text_input("Filter by Chapter Name", value="")
        selected_section = st.text_input("Filter by Section Name", value="")
        
        active_filters = {}
        if selected_doc != "All":
            active_filters["document"] = selected_doc
        if selected_page:
            try:
                active_filters["page"] = int(selected_page)
            except ValueError:
                active_filters["page"] = selected_page
        if selected_chapter:
            active_filters["chapter"] = selected_chapter
        if selected_section:
            active_filters["section"] = selected_section

    # Document uploader (supports PDF, DOCX, TXT, Markdown, HTML, Images)
    st.markdown("### 📁 Document Store")
    uploaded_files = st.file_uploader(
        "Ingest Files (Supports DOCX, PDF, Images, Markdown, HTML)",
        type=["pdf", "docx", "txt", "md", "html", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Check if uploader changed
        uploaded_names = [f.name for f in uploaded_files]
        # Compare with loaded engine files
        loaded_names = [d["source"] for d in st.session_state.rag_engine.documents]
        
        needs_ingestion = any(name not in loaded_names for name in uploaded_names)
        if needs_ingestion:
            with st.spinner("Ingesting new files into Vector DB..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in loaded_names:
                        continue
                        
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    
                    try:
                        progress_bar = st.sidebar.progress(0, text=f"Processing {uploaded_file.name}...")
                        
                        progress_bar.progress(15, text="Parsing layout & formatting contents...")
                        doc_id = st.session_state.rag_engine.load_document(tmp_path)
                        st.session_state.rag_engine.documents[doc_id]["source"] = uploaded_file.name
                        
                        progress_bar.progress(55, text="Extracting metadata and summaries...")
                        progress_bar.progress(75, text="Generating semantic chunks & scoring quality...")
                        num_ch = st.session_state.rag_engine.index_document(doc_id, chunk_size, chunk_overlap)
                        
                        progress_bar.progress(100, text="Indexing complete!")
                        st.sidebar.success(f"Successfully Indexed {uploaded_file.name} ({num_ch} semantic chunks)")
                        progress_bar.empty()
                    except Exception as e:
                        st.sidebar.error(f"Error indexing {uploaded_file.name}: {e}")
                    finally:
                        os.unlink(tmp_path)
                        
    # DB Stats
    total_docs = len(st.session_state.rag_engine.documents)
    total_chunks = len(st.session_state.rag_engine.chunks)
    st.markdown(f"**Indexed Docs**: `{total_docs}` | **Chunks**: `{total_chunks}`")

# Main Screen Split Layout: Chat Panel & Citations Panel
main_col, side_col = st.columns([3, 1])

with main_col:
    # Chat display container
    chat_container = st.container()
    
    with chat_container:
        # Display session messages
        for msg in st.session_state.memory.messages:
            if msg["role"] == "user":
                st.markdown(f"<div style='text-align: right;'><div class='user-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
            else:
                # Check for reasoning block inside thought process
                ans_text = msg["content"]
                thought = ""
                # Parse thought blocks if present
                if "[Reasoned via" in ans_text:
                    parts = ans_text.split("\n", 2)
                    if len(parts) >= 2:
                        thought = parts[0] + "\n" + parts[1]
                        ans_text = parts[2] if len(parts) > 2 else ""
                        
                with st.chat_message("assistant", avatar="🤖"):
                    if thought:
                        st.markdown(f"<div class='think-bubble'>{thought}</div>", unsafe_allow_html=True)
                    st.markdown(ans_text)
                    
    # Chat input box
    if prompt := st.chat_input("Query the knowledge platform..."):
        # Display user input immediately
        st.markdown(f"<div style='text-align: right;'><div class='user-bubble'>{prompt}</div></div>", unsafe_allow_html=True)
        st.session_state.memory.add_message("user", prompt)
        
        # Reset telemetry for the new query
        telemetry.reset()
        telemetry.start_span("total_query_latency")
        
        # 1. Query Routing & Intent Classification
        with st.spinner("🤖 Routing Query Strategy..."):
            hist_str = st.session_state.memory.get_history_string()
            routed = st.session_state.rag_engine.router.route_query(prompt, hist_str, active_filters)
            route = routed["route"]
            intent = routed["intent"]
            
        if route == "summarization":
            with st.spinner("📚 Generating hierarchical Map-Reduce summary..."):
                sum_mode = routed.get("summary_mode", "short")
                res = st.session_state.rag_engine.summarize_active_document(mode=sum_mode)
                summary_text = res.get("summary", "")
                
                answer = (
                    f"### 📄 Document Summary: **{res.get('document_name')}**\n"
                    f"- **Summary Type**: `{res.get('summary_type').upper()}`\n"
                    f"- **Pages Processed**: `{res.get('pages_processed')}`\n"
                    f"- **Analysis Confidence**: `{res.get('confidence')}`\n\n"
                    f"--- \n\n"
                    f"### Key Themes & Ideas\n"
                    f"{summary_text}"
                )
                telemetry.end_span("total_query_latency")
                
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(answer)
                st.session_state.memory.add_message("assistant", answer)
                st.rerun()
                
        elif route == "quote_extraction":
            with st.spinner("💬 Extracting key passage..."):
                quote_card = routed["data"]
                
                if quote_card.get("found", True):
                    answer = (
                        f"💬 **Verbatim Quote:**\n"
                        f"> \"{quote_card['quote']}\"\n\n"
                        f"**Contextual Explanation:**\n"
                        f"{quote_card['explanation']}\n\n"
                        f"**Source Document:** `{quote_card['source_document']}` | **Page/Section:** `{quote_card['page']}` | **Confidence:** `{quote_card['confidence']}`"
                    )
                else:
                    answer = (
                        f"⚠️ **Verbatim Passage:**\n"
                        f"> *\"{quote_card['quote']}\"*\n\n"
                        f"**Central Theme:**\n"
                        f"{quote_card['explanation']}"
                    )
                    
                telemetry.end_span("total_query_latency")
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(answer)
                st.session_state.memory.add_message("assistant", answer)
                st.rerun()
                
        else:
            # 2. Standard QA Retrieval Phase
            with st.status(f"🔍 Searching Context (Intent: {intent})") as status:
                telemetry.start_span("retrieval_and_filtering")
                final_chunks = routed["data"]
                retrieved_chunks = [(c, c.get("rrf_score", 1.0)) for c in final_chunks]
                telemetry.end_span("retrieval_and_filtering")
                
                # Execute Web Fallback if confidence is low, no chunks, or agent flagged it
                if web_search_fallback and not retrieved_chunks:
                    status.update(label="Confidence low. Triggering Web Search Fallback...", state="running")
                    telemetry.start_span("web_fallback")
                    web_hits = st.session_state.rag_engine.web_search.search(prompt, num_results=4)
                    for hit in web_hits:
                        retrieved_chunks.append((hit, 0.90))
                    telemetry.end_span("web_fallback")
                    
                status.update(label="Context query complete.", state="complete")
                
            # 3. Response Generation Phase
            with st.spinner("🤖 Formulating grounded response..."):
                telemetry.start_span("llm_response_generation")
                
                history_tuples = [(msg["role"], msg["content"]) for msg in st.session_state.memory.messages[:-1]]
            
            answer = st.session_state.rag_engine.generate_answer(
                query=prompt,
                context_chunks=retrieved_chunks,
                chat_history=history_tuples
            )
            telemetry.end_span("llm_response_generation")
            
        telemetry.end_span("total_query_latency")
        
        # Parse citations to display in the side panel
        chunks_only = [item[0] for item in retrieved_chunks]
        # Clean footnotes inside answer for clean preview
        _, citations = CitationEngine.extract_citations(answer, chunks_only)
        st.session_state.citations_list = citations
        
        # Display assistant response in chat bubble
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)
            
        st.session_state.memory.add_message("assistant", answer)
        st.rerun()

with side_col:
    st.markdown("### 📊 Observability & Citations")
    
    # Telemetry Panel
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Performance Logs")
    summary = telemetry.get_summary()
    latencies = summary.get("latencies_sec", {})
    
    col_l1, col_l2 = st.columns(2)
    with col_l1:
        st.metric("Total Query Time", f"{latencies.get('total_query_latency', 0.0):.2f}s")
        st.metric("Retrieval Time", f"{latencies.get('retrieval_and_filtering', 0.0):.2f}s")
    with col_l2:
        st.metric("Generation Time", f"{latencies.get('llm_response_generation', 0.0):.2f}s")
        if 'web_fallback' in latencies:
            st.metric("Web Search Time", f"{latencies.get('web_fallback', 0.0):.2f}s")
            
    # Token display
    tokens = summary.get("token_usage", {})
    st.markdown(f"**Total Tokens**: `{tokens.get('total_tokens', 0)}` | **Prompt**: `{tokens.get('prompt_tokens', 0)}` | **Completion**: `{tokens.get('completion_tokens', 0)}`")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Grounded Citations list
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("#### 🔍 Highlighted Citations")
    if st.session_state.citations_list:
        for cite in st.session_state.citations_list:
            st.markdown(f"""
            <div class='cite-side-card'>
                <strong>[{cite['index']}] {cite['source']}</strong> (Page {cite['page']})<br/>
                <span style='font-size: 0.8rem; color: #a5b4fc;'>Confidence: {cite['confidence_score']:.2f}</span><br/>
                <span style='font-size: 0.82rem; color: #94a3b8; font-style: italic;'>"{cite['snippet']}"</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No active citations. Ask a query to display source references.")
    st.markdown("</div>", unsafe_allow_html=True)
