
import streamlit as st

from src.rag.rag_pipeline import RAGPipeline


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Security Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# LOAD RAG PIPELINE
# --------------------------------------------------

@st.cache_resource
def load_pipeline():

    return RAGPipeline()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []

if "last_result" not in st.session_state:

    st.session_state.last_result = None

if "request_count" not in st.session_state:

    st.session_state.request_count = 0


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("🛡️ RAG Security Lab")

    st.caption(
        "Prompt Injection Defense Evaluation"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "AI Chatbot",
            "Attack Simulator",
            "Defense Comparison",
            "Analytics",
            "Documents"
        ]
    )

    st.divider()

    st.subheader("Configuration")

    st.info("D0 — Baseline RAG")

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    st.caption("Embedding: MiniLM-L6-v2")

    st.caption("Vector database: ChromaDB")

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.last_result = None

        st.session_state.request_count = 0

        st.rerun()


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.title("RAG Security Lab")

st.caption(
    "Evaluating the Generalization of "
    "Prompt-Injection Defenses in "
    "Retrieval-Augmented Generation Systems"
)


# --------------------------------------------------
# AI CHATBOT PAGE
# --------------------------------------------------

if page == "AI Chatbot":

    st.subheader(
        "Institutional Knowledge Assistant"
    )

    st.info(
        "Baseline mode: No additional experimental "
        "prompt-injection defenses are active."
    )

    result = st.session_state.last_result

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Defense Mode",
        "D0"
    )

    col2.metric(
        "Questions Asked",
        st.session_state.request_count
    )

    col3.metric(
        "Last Latency",
        (
            f"{result['total_latency_ms']:.0f} ms"
            if result
            else "—"
        )
    )

    col4.metric(
        "Last Token Usage",
        (
            result["total_tokens"]
            if result
            and result["total_tokens"] is not None
            else "—"
        )
    )

    st.divider()

    # Display conversation

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

    # Welcome message

    if not st.session_state.messages:

        with st.chat_message("assistant"):

            st.markdown(
                "Hello! I am your institutional "
                "knowledge assistant. Ask me about "
                "academics, attendance, examinations, "
                "placements, library services, hostel "
                "policies or administration."
            )

    question = st.chat_input(
        "Ask about institutional policies..."
    )

    if question:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        st.session_state.last_result = None

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "Retrieving knowledge and "
                    "generating response..."
                ):

                    pipeline = load_pipeline()

                    result = pipeline.ask(
                        question=question,
                        top_k=top_k
                    )

                answer = result["answer"]

                if answer.strip():

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                else:

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            "The model returned "
                            "an empty response."
                        )
                    })

                st.session_state.last_result = result

                st.session_state.request_count += 1

                st.rerun()

            except Exception as error:

                st.error(
                    f"Request failed: "
                    f"{type(error).__name__}. "
                    "Check your API configuration "
                    "and connection."
                )

    # Response diagnostics

    if st.session_state.last_result:

        result = st.session_state.last_result

        st.divider()

        st.subheader("Response Diagnostics")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Retrieval Latency",
            f"{result['retrieval_latency_ms']} ms"
        )

        c2.metric(
            "Generation Latency",
            f"{result['generation_latency_ms']} ms"
        )

        c3.metric(
            "Total Latency",
            f"{result['total_latency_ms']} ms"
        )

        c4, c5, c6 = st.columns(3)

        c4.metric(
            "Input Tokens",
            (
                result["input_tokens"]
                if result["input_tokens"] is not None
                else "Unavailable"
            )
        )

        c5.metric(
            "Output Tokens",
            (
                result["output_tokens"]
                if result["output_tokens"] is not None
                else "Unavailable"
            )
        )

        c6.metric(
            "Retrieved Chunks",
            len(result["retrieved_chunks"])
        )

        st.caption(
            f"Model: {result['model']} | "
            f"Defense: {result['defense_mode']} | "
            f"Status: {result['execution_status']}"
        )

        # Retrieved sources

        with st.expander(
            "Retrieved Source Documents",
            expanded=False
        ):

            for chunk in result["retrieved_chunks"]:

                st.markdown(
                    f"### Rank {chunk['rank']} — "
                    f"{chunk['document_id']}"
                )

                st.caption(
                    f"Domain: {chunk['domain']} | "
                    f"Similarity: "
                    f"{chunk['similarity']:.4f}"
                )

                st.write(chunk["text"])

                st.caption(
                    f"Source: {chunk['source']}"
                )

                st.divider()

        # Raw result for development

        with st.expander("Raw Response Metadata"):

            st.json(result)


# --------------------------------------------------
# ATTACK SIMULATOR
# --------------------------------------------------

elif page == "Attack Simulator":

    st.subheader("Attack Simulator")

    st.info(
        "The controlled prompt-injection testing "
        "environment will be implemented in the "
        "upcoming defense development stage."
    )


# --------------------------------------------------
# DEFENSE COMPARISON
# --------------------------------------------------

elif page == "Defense Comparison":

    st.subheader("Defense Comparison")

    st.info(
        "Side-by-side comparison of D0–D5 "
        "will be available after defense integration."
    )


# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

elif page == "Analytics":

    st.subheader("Research Analytics")

    st.info(
        "Attack success rate, detection recall, "
        "false positive rate, latency and "
        "generalization metrics will appear here "
        "after the evaluation engine is implemented."
    )


# --------------------------------------------------
# DOCUMENTS
# --------------------------------------------------

elif page == "Documents":

    st.subheader("Institutional Knowledge Corpus")

    try:

        from src.rag.document_loader import (
            load_documents
        )

        documents = load_documents()

        st.metric(
            "Total Documents",
            len(documents)
        )

        domains = sorted({
            document.domain
            for document in documents
        })

        selected_domain = st.selectbox(
            "Filter by domain",
            ["All"] + domains
        )

        filtered_documents = [
            document
            for document in documents
            if selected_domain == "All"
            or document.domain == selected_domain
        ]

        st.caption(
            f"Showing {len(filtered_documents)} "
            f"documents"
        )

        for document in filtered_documents:

            with st.expander(
                f"{document.document_id} — "
                f"{document.domain}"
            ):

                st.write(document.text)

                st.caption(document.source)

    except Exception as error:

        st.error(
            f"Unable to load documents: "
            f"{type(error).__name__}"
        )
