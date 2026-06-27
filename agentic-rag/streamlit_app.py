import warnings
import sys

from dotenv import load_dotenv

try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError as exc:
    print(
        "\nStreamlit cannot start because installed package versions are incompatible.\n\n"
        "Fix it with:\n\n"
        '  cd "C:\\Users\\USER\\OneDrive\\Desktop\\chief raw"\n'
        "  python -m pip install --upgrade --force-reinstall -r requirements.txt\n\n"
        f"Original error: {exc}\n"
    )
    sys.exit(0)


if get_script_run_ctx() is None:
    print(
        "\nThis is a Streamlit app. Run it with:\n\n"
        '  cd "C:\\Users\\USER\\OneDrive\\Desktop\\chief raw\\agentic-rag"\n'
        "  python -m streamlit run streamlit_app.py\n"
    )
    sys.exit(0)

from app.agents.document_summarizer import summarize_documents
from app.graph.workflow import run_agentic_rag
from app.rag.ingestion import ingest_documents
from app.rag.retriever import list_document_sources


warnings.filterwarnings("ignore")
load_dotenv()

st.set_page_config(page_title="Agentic RAG", layout="wide")
st.title("Intelligent Document-Based Agentic RAG System")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "document_summary" not in st.session_state:
    st.session_state["document_summary"] = ""
if "available_sources" not in st.session_state:
    st.session_state["available_sources"] = []

if not st.session_state["available_sources"]:
    try:
        st.session_state["available_sources"] = list_document_sources()
    except Exception:
        st.session_state["available_sources"] = []

with st.sidebar:
    st.subheader("Mode")
    mode = st.radio("Select Mode", ["Agentic RAG", "Standard RAG"])

    if st.button("Refresh Document List", use_container_width=True):
        try:
            st.session_state["available_sources"] = list_document_sources()
        except Exception as exc:
            st.warning("Could not refresh document list.")
            st.caption(str(exc))

    source_options = ["All documents", *st.session_state["available_sources"]]
    selected_source_label = st.selectbox(
        "Chat Scope",
        source_options,
        help="Choose one document to chat with only that file, or use all indexed documents.",
    )
    selected_source = None if selected_source_label == "All documents" else selected_source_label

    uploaded_files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=["pdf", "doc", "docx", "txt", "pptx", "xlsx"],
    )
    urls = st.text_area("Enter URLs", placeholder="https://example.com").splitlines()

    if st.button("Ingest Documents", use_container_width=True):
        with st.spinner("Indexing documents..."):
            try:
                ingest_result = ingest_documents(uploaded_files, urls, return_chunks=True)
                count = ingest_result["count"]
            except Exception as exc:
                st.error(
                    "Document ingestion could not start because the local embedding "
                    "dependencies are not installed correctly. Run this in PowerShell:\n\n"
                    "python -m pip install --upgrade --force-reinstall -r requirements.txt"
                )
                st.caption(str(exc))
                st.stop()
        if count:
            st.success(f"Indexed {count} chunks.")
            try:
                st.session_state["available_sources"] = list_document_sources()
            except Exception:
                pass

            with st.spinner("Creating document summary..."):
                try:
                    st.session_state["document_summary"] = summarize_documents(
                        ingest_result["chunks"]
                    )
                except Exception as exc:
                    st.session_state["document_summary"] = (
                        "The documents were indexed, but summary generation failed. "
                        f"Reason: {exc}"
                    )
        else:
            st.warning("No supported documents or URLs were found.")

if st.session_state["document_summary"]:
    with st.expander("Document Summary Mode", expanded=True):
        st.markdown(st.session_state["document_summary"])

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask your question")

if query:
    st.session_state["chat_history"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking through the documents..."):
            try:
                result = run_agentic_rag(
                    query,
                    force_single_pass=(mode == "Standard RAG"),
                    selected_source=selected_source,
                )
                answer = result["answer"]
                sources = result["sources"]
                confidence = result["confidence"]
                details = result
            except Exception as exc:
                st.error(
                    "The agent could not run because one of the local dependencies "
                    "is not installed correctly. Run this in PowerShell:\n\n"
                    "python -m pip install --upgrade --force-reinstall -r requirements.txt"
                )
                st.caption(str(exc))
                st.stop()

        st.markdown("### Answer")
        st.write(answer)

        if sources:
            st.markdown("### Sources")
            for source in sources:
                st.write(f"- {source}")

        with st.expander("Multi-Agent Trace View", expanded=False):
            st.write(f"Mode: {mode}")
            st.write(f"Chat scope: {details.get('selected_source') or 'All documents'}")
            st.write(f"Confidence: {confidence:.2f}")
            st.write(f"Retrieved chunks: {details.get('retrieved_count', 0)}")

            st.markdown("**Planner Decision**")
            st.write(details.get("plan", ""))

            st.markdown("**Rewritten Query**")
            st.write(details.get("refined_query", ""))

            st.markdown("**Retrieved Sources**")
            if sources:
                for source in sources:
                    st.write(f"- {source}")
            else:
                st.write("No sources retrieved.")

            st.markdown("**Verifier Result**")
            st.write(details.get("verification", ""))

            st.markdown("**Reflection Attempts**")
            reflection_attempts = details.get("reflection_attempts", [])
            if reflection_attempts:
                for index, attempt in enumerate(reflection_attempts, start=1):
                    st.write(f"{index}. {attempt}")
            else:
                st.write("No reflection retry was needed.")

            st.markdown("**Final Reasoning Summary**")
            st.write(details.get("final_reasoning_summary", ""))

    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
