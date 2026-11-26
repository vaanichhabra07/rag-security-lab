
import streamlit as st

from src.rag.rag_pipeline import RAGPipeline
from src.rag.document_loader import load_documents


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="RAG Security Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# DEFENSE CONFIGURATIONS
# ==================================================

DEFENSE_MODES = {
    "D0": "D0 — Baseline RAG",
    "D1": "D1 — Rule-Based Defense"
}


# ==================================================
# CACHED RESOURCES
# ==================================================

@st.cache_resource
def load_pipeline():

    return RAGPipeline()


@st.cache_data
def load_corpus():

    return load_documents()


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:

    st.session_state.messages = []

if "last_result" not in st.session_state:

    st.session_state.last_result = None

if "request_count" not in st.session_state:

    st.session_state.request_count = 0

if "detected_count" not in st.session_state:

    st.session_state.detected_count = 0

if "blocked_count" not in st.session_state:

    st.session_state.blocked_count = 0


# ==================================================
# SIDEBAR
# ==================================================

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

    defense_mode = st.selectbox(
        "Defense Configuration",
        options=list(DEFENSE_MODES.keys()),
        format_func=lambda mode: DEFENSE_MODES[mode]
    )

    if defense_mode == "D0":

        st.warning(
            "Baseline mode: Experimental "
            "defenses disabled."
        )

    elif defense_mode == "D1":

        st.success(
            "Rule-based input and context "
            "scanning enabled."
        )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=3
    )

    st.divider()

    st.subheader("System Information")

    st.caption("Embedding: MiniLM-L6-v2")

    st.caption("Vector database: ChromaDB")

    st.caption("LLM provider: Groq (current configuration)")

    st.divider()

    st.subheader("Session Statistics")

    st.metric(
        "Processed Requests",
        st.session_state.request_count
    )

    st.metric(
        "Suspicious Requests",
        st.session_state.detected_count
    )

    st.metric(
        "Blocked Requests",
        st.session_state.blocked_count
    )

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.last_result = None

        st.session_state.request_count = 0

        st.session_state.detected_count = 0

        st.session_state.blocked_count = 0

        st.rerun()


# ==================================================
# MAIN HEADER
# ==================================================

st.title("RAG Security Lab")

st.caption(
    "Evaluating the Generalization of "
    "Prompt-Injection Defenses in "
    "Retrieval-Augmented Generation Systems"
)


# ==================================================
# AI CHATBOT PAGE
# ==================================================

if page == "AI Chatbot":

    st.subheader(
        "Institutional Knowledge Assistant"
    )

    if defense_mode == "D0":

        st.info(
            "D0 — Baseline RAG: No additional "
            "experimental prompt-injection "
            "defenses are active."
        )

    elif defense_mode == "D1":

        st.success(
            "D1 — Rule-Based Defense: "
            "User inputs and retrieved context "
            "are scanned for suspicious instructions."
        )

    result = st.session_state.last_result

    # ----------------------------------------------
    # TOP METRICS
    # ----------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Selected Defense",
        defense_mode
    )

    col2.metric(
        "Processed Requests",
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

    # ----------------------------------------------
    # CONVERSATION HISTORY
    # ----------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("defense_mode"):

                st.caption(
                    "Defense: "
                    + message["defense_mode"]
                )

            if message.get("blocked"):

                st.warning(
                    "Request blocked by security defense."
                )

    # ----------------------------------------------
    # WELCOME MESSAGE
    # ----------------------------------------------

    if not st.session_state.messages:

        with st.chat_message("assistant"):

            st.markdown(
                "Hello! I am your institutional "
                "knowledge assistant.\n\n"

                "You can ask me about:\n"

                "- Academics and attendance\n"
                "- Examinations and placements\n"
                "- Library and hostel services\n"
                "- Admissions and administration\n"
                "- Scholarships and student support\n\n"

                "Use the defense configuration "
                "in the sidebar to switch between "
                "baseline and protected modes."
            )

    # ----------------------------------------------
    # CHAT INPUT
    # ----------------------------------------------

    question = st.chat_input(
        "Ask about institutional policies..."
    )

    if question and question.strip():

        question = question.strip()

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
                    "Processing request, retrieving "
                    "knowledge and applying defenses..."
                ):

                    pipeline = load_pipeline()

                    result = pipeline.ask(
                        question=question,
                        top_k=top_k,
                        defense_mode=defense_mode
                    )

                answer = result.get("answer", "")

                if not answer.strip():

                    answer = (
                        "The model returned "
                        "an empty response."
                    )

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "defense_mode":
                        result["defense_mode"],

                    "blocked":
                        result.get("blocked", False),

                    "attack_detected":
                        result.get(
                            "attack_detected", False
                        )
                })

                st.session_state.last_result = result

                st.session_state.request_count += 1

                if result.get("attack_detected"):

                    st.session_state.detected_count += 1

                if result.get("blocked"):

                    st.session_state.blocked_count += 1

                st.rerun()

            except Exception as error:

                st.error(
                    f"Request failed: "
                    f"{type(error).__name__}. "
                    "Check your API configuration, "
                    "connection and backend."
                )

                st.caption(
                    "Run the backend testing scripts "
                    "to diagnose the error."
                )

    # ==============================================
    # RESPONSE DIAGNOSTICS
    # ==============================================

    if st.session_state.last_result:

        result = st.session_state.last_result

        st.divider()

        st.subheader("Response Diagnostics")

        st.caption(
            f"Model: {result['model']} | "
            f"Defense used: {result['defense_mode']} | "
            f"Status: {result['execution_status']}"
        )

        # ------------------------------------------
        # LATENCY METRICS
        # ------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Retrieval Latency",
            f"{result['retrieval_latency_ms']:.2f} ms"
        )

        c2.metric(
            "Generation Latency",
            f"{result['generation_latency_ms']:.2f} ms"
        )

        c3.metric(
            "Defense Latency",
            f"{result.get('defense_latency_ms', 0):.4f} ms"
        )

        c4.metric(
            "Total Latency",
            f"{result['total_latency_ms']:.2f} ms"
        )

        # ------------------------------------------
        # TOKEN METRICS
        # ------------------------------------------

        c5, c6, c7, c8 = st.columns(4)

        c5.metric(
            "Input Tokens",
            (
                result["input_tokens"]
                if result["input_tokens"] is not None
                else "Unavailable"
            )
        )

        c6.metric(
            "Output Tokens",
            (
                result["output_tokens"]
                if result["output_tokens"] is not None
                else "Unavailable"
            )
        )

        c7.metric(
            "Total Tokens",
            (
                result["total_tokens"]
                if result["total_tokens"] is not None
                else "Unavailable"
            )
        )

        c8.metric(
            "Retrieved Chunks",
            len(result.get("retrieved_chunks", []))
        )

        # ==========================================
        # SECURITY DIAGNOSTICS
        # ==========================================

        st.divider()

        st.subheader("Security Diagnostics")

        detection_active = (
            result["defense_mode"] != "D0"
        )

        attack_detected = result.get(
            "attack_detected", False
        )

        blocked = result.get("blocked", False)

        llm_called = result.get("llm_called", False)

        flagged_chunks = result.get(
            "flagged_chunks", []
        )

        matched_rules = result.get(
            "matched_rules", []
        )

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Attack Detection",
            (
                "Not evaluated"
                if not detection_active
                else (
                    "Detected"
                    if attack_detected
                    else "Not detected"
                )
            )
        )

        s2.metric(
            "Request Blocked",
            "Yes" if blocked else "No"
        )

        s3.metric(
            "Filtered Chunks",
            len(flagged_chunks)
        )

        s4.metric(
            "LLM Called",
            "Yes" if llm_called else "No"
        )

        # ------------------------------------------
        # SECURITY STATUS
        # ------------------------------------------

        if not detection_active:

            st.info(
                "D0 does not perform application-level "
                "attack detection. An undetected attack "
                "does not mean the request is safe."
            )

        elif blocked:

            st.error(
                "The request was blocked by "
                "the active security defense."
            )

        elif attack_detected:

            st.warning(
                "Suspicious content was detected. "
                "Flagged context chunks were removed "
                "before generation."
            )

        else:

            st.success(
                "No suspicious content was detected "
                "by the configured security rules."
            )

        # ------------------------------------------
        # MATCHED SECURITY RULES
        # ------------------------------------------

        if matched_rules:

            with st.expander(
                "Matched Security Rules",
                expanded=True
            ):

                for rule in matched_rules:

                    st.markdown(
                        f"**Rule:** "
                        f"{rule.get('rule_id', 'Unknown')}"
                    )

                    st.write(
                        "Category:",
                        rule.get("category", "Unknown")
                    )

                    severity = rule.get(
                        "severity", "unknown"
                    )

                    if severity == "high":

                        st.error("Severity: HIGH")

                    elif severity == "medium":

                        st.warning("Severity: MEDIUM")

                    else:

                        st.info(
                            f"Severity: {severity.upper()}"
                        )

                    st.divider()

        # ------------------------------------------
        # FILTERED CONTEXT
        # ------------------------------------------

        if flagged_chunks:

            with st.expander(
                "Filtered Context Chunks",
                expanded=False
            ):

                st.warning(
                    "These chunks were excluded "
                    "from the generation context."
                )

                for chunk in flagged_chunks:

                    st.markdown(
                        f"**Document:** "
                        f"{chunk['document_id']}"
                    )

                    st.caption(
                        f"Chunk: {chunk['chunk_id']}"
                    )

                    st.write(chunk["text"])

                    st.divider()

        # ------------------------------------------
        # DEFENSE EXECUTION DETAILS
        # ------------------------------------------

        if result.get("defense_details"):

            with st.expander(
                "Defense Execution Details"
            ):

                st.json(
                    result["defense_details"]
                )

        # ==========================================
        # RETRIEVED DOCUMENTS
        # ==========================================

        st.divider()

        st.subheader("Knowledge Retrieval")

        retrieved_chunks = result.get(
            "retrieved_chunks", []
        )

        context_used = result.get(
            "context_used", []
        )

        used_chunk_ids = {
            chunk["chunk_id"]
            for chunk in context_used
        }

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Retrieved Chunks",
            len(retrieved_chunks)
        )

        r2.metric(
            "Context Chunks Used",
            len(context_used)
        )

        r3.metric(
            "Excluded Chunks",
            len(retrieved_chunks) - len(context_used)
        )

        with st.expander(
            "Retrieved Source Documents",
            expanded=False
        ):

            if not retrieved_chunks:

                st.info(
                    "No documents were retrieved "
                    "for this request."
                )

            for chunk in retrieved_chunks:

                st.markdown(
                    f"### Rank {chunk['rank']} — "
                    f"{chunk['document_id']}"
                )

                st.caption(
                    f"Domain: {chunk['domain']} | "
                    f"Similarity: "
                    f"{chunk['similarity']:.4f}"
                )

                if chunk["chunk_id"] in used_chunk_ids:

                    st.success(
                        "Included in generation context"
                    )

                else:

                    st.warning(
                        "Excluded from generation context"
                    )

                st.write(chunk["text"])

                st.caption(
                    f"Source: {chunk['source']}"
                )

                st.divider()

        # ------------------------------------------
        # RAW RESPONSE METADATA
        # ------------------------------------------

        with st.expander(
            "Raw Response Metadata"
        ):

            st.json(result)


# ==================================================
# ATTACK SIMULATOR PAGE
# ==================================================

elif page == "Attack Simulator":

    st.subheader("Attack Simulator")

    st.info(
        "The controlled attack simulator will "
        "be implemented after the remaining "
        "defense mechanisms are integrated."
    )

    st.markdown(
        "### Planned Attack Families"
    )

    st.markdown(
        """
        - Direct prompt injection
        - Instruction manipulation
        - Obfuscated prompt injection
        - Indirect prompt injection
        - Multi-stage prompt injection
        """
    )


# ==================================================
# DEFENSE COMPARISON PAGE
# ==================================================

elif page == "Defense Comparison":

    st.subheader("Defense Comparison")

    st.info(
        "Side-by-side experimental comparison "
        "will be implemented after D0–D5 "
        "integration."
    )

    defense_table = [

        {
            "Configuration": "D0",
            "Defense": "Baseline RAG",
            "Status": "Implemented"
        },

        {
            "Configuration": "D1",
            "Defense": "Rule-Based Detection",
            "Status": "Implemented"
        },

        {
            "Configuration": "D2",
            "Defense": "ML/LLM Detection",
            "Status": "Pending"
        },

        {
            "Configuration": "D3",
            "Defense": "Context Isolation",
            "Status": "Pending"
        },

        {
            "Configuration": "D4",
            "Defense": "Output Validation",
            "Status": "Pending"
        },

        {
            "Configuration": "D5",
            "Defense": "Combined Defense",
            "Status": "Pending"
        }
    ]

    st.dataframe(
        defense_table,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# ANALYTICS PAGE
# ==================================================

elif page == "Analytics":

    st.subheader("Research Analytics")

    st.caption(
        "Current session statistics — "
        "not formal benchmark results."
    )

    a1, a2, a3 = st.columns(3)

    a1.metric(
        "Processed Requests",
        st.session_state.request_count
    )

    a2.metric(
        "Suspicious Requests",
        st.session_state.detected_count
    )

    a3.metric(
        "Blocked Requests",
        st.session_state.blocked_count
    )

    st.divider()

    st.markdown("### Planned Research Metrics")

    st.markdown(
        """
        - Attack success rate (ASR)
        - Detection precision
        - Detection recall
        - False positive rate
        - F1 score
        - Answer utility
        - Grounding accuracy
        - Seen versus unseen attack performance
        - Generalization gap
        - Defense latency
        - Token consumption
        - API cost
        """
    )

    st.info(
        "Formal research metrics will appear "
        "after the automated evaluation engine "
        "and benchmark experiments are completed."
    )


# ==================================================
# DOCUMENTS PAGE
# ==================================================

elif page == "Documents":

    st.subheader("Institutional Knowledge Corpus")

    try:

        documents = load_corpus()

        domains = sorted({
            document.domain
            for document in documents
        })

        total_words = sum(
            len(document.text.split())
            for document in documents
        )

        d1, d2, d3 = st.columns(3)

        d1.metric(
            "Total Documents",
            len(documents)
        )

        d2.metric(
            "Knowledge Domains",
            len(domains)
        )

        d3.metric(
            "Total Words",
            f"{total_words:,}"
        )

        st.divider()

        selected_domain = st.selectbox(
            "Filter by Domain",
            ["All"] + domains
        )

        search_query = st.text_input(
            "Search Documents",
            placeholder=(
                "Search by document ID, "
                "domain or content..."
            )
        )

        filtered_documents = []

        for document in documents:

            domain_matches = (
                selected_domain == "All"
                or document.domain == selected_domain
            )

            searchable_text = (
                document.document_id
                + " "
                + document.domain
                + " "
                + document.text
            ).casefold()

            search_matches = (
                not search_query.strip()
                or search_query.strip().casefold()
                in searchable_text
            )

            if domain_matches and search_matches:

                filtered_documents.append(document)

        st.caption(
            f"Showing {len(filtered_documents)} "
            f"of {len(documents)} documents"
        )

        if not filtered_documents:

            st.warning(
                "No documents match your filters."
            )

        for document in filtered_documents:

            with st.expander(
                f"{document.document_id} — "
                f"{document.domain.title()}"
            ):

                st.caption(
                    f"Domain: {document.domain}"
                )

                st.caption(
                    f"Word count: "
                    f"{len(document.text.split())}"
                )

                st.write(document.text)

                st.caption(
                    f"Source: {document.source}"
                )

    except Exception as error:

        st.error(
            f"Unable to load institutional documents: "
            f"{type(error).__name__}"
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "RAG Security Lab | Institutional RAG "
    "Security Research | B.Tech Major Project"
)
