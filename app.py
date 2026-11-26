
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
    "D1": "D1 — Rule-Based Defense",
    "D2": "D2 — LLM-Based Detection"
}

DEFENSE_DESCRIPTIONS = {
    "D0": (
        "Baseline RAG: No additional experimental "
        "prompt-injection defenses are active."
    ),
    "D1": (
        "Rule-Based Defense: User inputs and retrieved "
        "context are scanned using predefined security rules."
    ),
    "D2": (
        "LLM-Based Detection: User inputs and retrieved "
        "context are evaluated by an AI security classifier."
    )
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

SESSION_DEFAULTS = {
    "messages": [],
    "last_result": None,
    "request_count": 0,
    "detected_count": 0,
    "blocked_count": 0,
    "classifier_error_count": 0,
    "request_history": []
}

for key, value in SESSION_DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def format_tokens(value):

    if value is None:
        return "Unavailable"

    return f"{value:,}"


def format_latency(value):

    if value is None:
        return "Unavailable"

    return f"{value:.2f} ms"


def reset_session():

    st.session_state.messages = []

    st.session_state.last_result = None

    st.session_state.request_count = 0

    st.session_state.detected_count = 0

    st.session_state.blocked_count = 0

    st.session_state.classifier_error_count = 0

    st.session_state.request_history = []


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
            "Experimental defenses disabled."
        )

    elif defense_mode == "D1":

        st.success(
            "Rule-based input and context "
            "scanning enabled."
        )

    elif defense_mode == "D2":

        st.success(
            "LLM-based semantic security "
            "classification enabled."
        )

        st.caption(
            "D2 requires additional API requests "
            "and may increase response latency."
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

    st.caption("Current API configuration: Groq")

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

    st.metric(
        "Classifier Errors",
        st.session_state.classifier_error_count
    )

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        reset_session()

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
            DEFENSE_DESCRIPTIONS[defense_mode]
        )

    else:

        st.success(
            DEFENSE_DESCRIPTIONS[defense_mode]
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
            format_latency(
                result["total_latency_ms"]
            )
            if result
            else "—"
        )
    )

    col4.metric(
        "Last Combined Tokens",
        (
            format_tokens(
                result.get(
                    "combined_total_tokens",
                    result.get("total_tokens")
                )
            )
            if result
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

            if message.get("execution_status") == "defense_error":

                st.error(
                    "The security classifier "
                    "encountered an error."
                )

            elif message.get("blocked"):

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
                "D0, D1 and D2."
            )

            st.caption(
                "Questions are currently processed "
                "independently of previous messages."
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
                        ),

                    "execution_status":
                        result.get("execution_status")
                })

                st.session_state.last_result = result

                st.session_state.request_count += 1

                if result.get("attack_detected"):

                    st.session_state.detected_count += 1

                if result.get("blocked"):

                    st.session_state.blocked_count += 1

                st.session_state.classifier_error_count += (
                    result.get("classifier_errors", 0)
                )

                st.session_state.request_history.append({

                    "Defense": result["defense_mode"],

                    "Model": result["model"],

                    "Detected": (
                        result.get("attack_detected", False)
                        if result["defense_mode"] != "D0"
                        else None
                    ),

                    "Blocked":
                        result.get("blocked", False),

                    "Latency (ms)":
                        result.get("total_latency_ms"),

                    "Generation Tokens":
                        result.get("total_tokens"),

                    "Defense Tokens":
                        result.get("defense_total_tokens", 0),

                    "Combined Tokens":
                        result.get(
                            "combined_total_tokens",
                            result.get("total_tokens")
                        ),

                    "Classifier Errors":
                        result.get("classifier_errors", 0),

                    "Status":
                        result.get("execution_status")
                })

                st.rerun()

            except Exception as error:

                error_message = (
                    f"Request failed: "
                    f"{type(error).__name__}. "
                    "Check your API configuration, "
                    "connection and backend."
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "defense_mode": defense_mode,
                    "execution_status": "error"
                })

                st.rerun()

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

        if result["defense_mode"] != defense_mode:

            st.info(
                "These diagnostics belong to the previous "
                "request and its selected defense."
            )

        # ------------------------------------------
        # LATENCY METRICS
        # ------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Retrieval Latency",
            format_latency(
                result.get("retrieval_latency_ms")
            )
        )

        c2.metric(
            "Generation Latency",
            format_latency(
                result.get("generation_latency_ms")
            )
        )

        c3.metric(
            "Defense Latency",
            format_latency(
                result.get("defense_latency_ms", 0)
            )
        )

        c4.metric(
            "Total Latency",
            format_latency(
                result.get("total_latency_ms")
            )
        )

        # ------------------------------------------
        # TOKEN METRICS
        # ------------------------------------------

        c5, c6, c7, c8 = st.columns(4)

        c5.metric(
            "Generation Input Tokens",
            format_tokens(
                result.get("input_tokens")
            )
        )

        c6.metric(
            "Generation Output Tokens",
            format_tokens(
                result.get("output_tokens")
            )
        )

        c7.metric(
            "Generation Total Tokens",
            format_tokens(
                result.get("total_tokens")
            )
        )

        c8.metric(
            "Combined Tokens",
            format_tokens(
                result.get(
                    "combined_total_tokens",
                    result.get("total_tokens")
                )
            )
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

        classifier_errors = result.get(
            "classifier_errors", 0
        )

        flagged_chunks = result.get(
            "flagged_chunks", []
        )

        matched_rules = result.get(
            "matched_rules", []
        )

        if not detection_active:

            detection_status = "Not evaluated"

        elif attack_detected:

            detection_status = "Detected"

        elif classifier_errors:

            detection_status = "Incomplete"

        else:

            detection_status = "Not detected"

        s1, s2, s3, s4 = st.columns(4)

        s1.metric(
            "Attack Detection",
            detection_status
        )

        s2.metric(
            "Request Blocked",
            "Yes" if blocked else "No"
        )

        s3.metric(
            "Excluded Chunks",
            len(flagged_chunks)
        )

        s4.metric(
            "Answer LLM Called",
            "Yes" if llm_called else "No"
        )

        # ------------------------------------------
        # SECURITY STATUS
        # ------------------------------------------

        if not detection_active:

            st.info(
                "D0 does not perform application-level "
                "attack detection."
            )

        elif result.get("execution_status") == "defense_error":

            st.error(
                "The request was blocked because "
                "security classification was unavailable."
            )

        elif blocked:

            st.error(
                "The request was blocked by "
                "the active security defense."
            )

        elif classifier_errors:

            st.warning(
                "Security classification was incomplete. "
                "Some classifier requests failed."
            )

        elif attack_detected:

            st.warning(
                "Suspicious content was detected. "
                "Flagged context chunks were excluded "
                "before generation."
            )

        else:

            st.success(
                "No suspicious content was detected "
                "by the configured defense."
            )

        # ------------------------------------------
        # D1 MATCHED SECURITY RULES
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

        # ==========================================
        # D2 LLM CLASSIFIER DIAGNOSTICS
        # ==========================================

        if result.get("defense_mode") == "D2":

            st.divider()

            st.subheader(
                "LLM Classifier Diagnostics"
            )

            classifier_results = result.get(
                "classifier_results", []
            )

            benign_count = sum(
                scan.get("label") == "BENIGN"
                for scan in classifier_results
            )

            adversarial_count = sum(
                scan.get("label") == "ADVERSARIAL"
                for scan in classifier_results
            )

            unknown_count = sum(
                scan.get("label") == "UNKNOWN"
                for scan in classifier_results
            )

            st.caption(
                "Classifier model: "
                + str(
                    result.get(
                        "classifier_model",
                        "Unavailable"
                    )
                )
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Classifier Requests",
                result.get(
                    "defense_api_calls", 0
                )
            )

            m2.metric(
                "Benign",
                benign_count
            )

            m3.metric(
                "Adversarial",
                adversarial_count
            )

            m4.metric(
                "Unknown",
                unknown_count
            )

            t1, t2, t3, t4 = st.columns(4)

            t1.metric(
                "Defense Input Tokens",
                format_tokens(
                    result.get("defense_input_tokens")
                )
            )

            t2.metric(
                "Defense Output Tokens",
                format_tokens(
                    result.get("defense_output_tokens")
                )
            )

            t3.metric(
                "Defense Total Tokens",
                format_tokens(
                    result.get("defense_total_tokens")
                )
            )

            t4.metric(
                "Classifier Errors",
                classifier_errors
            )

            if classifier_errors:

                st.warning(
                    "One or more classifier requests "
                    "failed. UNKNOWN classifications "
                    "must not be counted as successful "
                    "attack detections."
                )

            st.markdown(
                "#### Individual Classification Results"
            )

            for index, scan in enumerate(
                classifier_results, 1
            ):

                channel = scan.get(
                    "channel", "unknown"
                )

                label = scan.get(
                    "label", "UNKNOWN"
                )

                with st.expander(
                    f"Classification {index} — "
                    f"{channel.replace('_', ' ').title()} "
                    f"— {label}",
                    expanded=(index == 1)
                ):

                    if label == "BENIGN":

                        st.success("Label: BENIGN")

                    elif label == "ADVERSARIAL":

                        st.error("Label: ADVERSARIAL")

                    else:

                        st.warning("Label: UNKNOWN")

                    if scan.get("document_id"):

                        st.write(
                            "Document:",
                            scan["document_id"]
                        )

                    if scan.get("chunk_id"):

                        st.write(
                            "Chunk:",
                            scan["chunk_id"]
                        )

                    st.write(
                        "Category:",
                        scan.get("category", "Unknown")
                    )

                    st.write(
                        "Reason:",
                        scan.get("reason", "Unavailable")
                    )

                    st.write(
                        "Action:",
                        scan.get("action", "Unknown")
                    )

                    st.write(
                        "Classification status:",
                        scan.get("classification_status")
                    )

                    st.write(
                        "Detection latency:",
                        format_latency(
                            scan.get("detection_latency_ms")
                        )
                    )

                    st.write(
                        "Tokens:",
                        format_tokens(
                            scan.get("total_tokens")
                        )
                    )

                    if scan.get("error_type"):

                        st.error(
                            "Error type: "
                            + scan["error_type"]
                        )

                    st.json(scan)

        # ------------------------------------------
        # FILTERED CONTEXT
        # ------------------------------------------

        if flagged_chunks:

            with st.expander(
                "Excluded Context Chunks",
                expanded=False
            ):

                st.warning(
                    "These chunks were excluded "
                    "by the active defense."
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

    st.markdown("### Planned Attack Families")

    st.markdown(
        """
        - Direct prompt injection
        - Instruction manipulation
        - Obfuscated prompt injection
        - Indirect prompt injection
        - Multi-stage prompt injection
        """
    )

    st.markdown("### Available Defense Configurations")

    for mode, description in DEFENSE_MODES.items():

        st.write(description)


# ==================================================
# DEFENSE COMPARISON PAGE
# ==================================================

elif page == "Defense Comparison":

    st.subheader("Defense Comparison")

    st.info(
        "Full side-by-side experimental comparison "
        "will be implemented after D0–D5 integration."
    )

    defense_table = [

        {
            "Configuration": "D0",
            "Defense": "Baseline RAG",
            "Technique": "No experimental defense",
            "Status": "Implemented"
        },

        {
            "Configuration": "D1",
            "Defense": "Rule-Based Detection",
            "Technique": "Regex and text normalization",
            "Status": "Implemented"
        },

        {
            "Configuration": "D2",
            "Defense": "LLM-Based Detection",
            "Technique": "Prompted binary classification",
            "Status": "Implemented"
        },

        {
            "Configuration": "D3",
            "Defense": "Context Isolation",
            "Technique": "Trust-boundary separation",
            "Status": "Pending"
        },

        {
            "Configuration": "D4",
            "Defense": "Output Validation",
            "Technique": "Generated response validation",
            "Status": "Pending"
        },

        {
            "Configuration": "D5",
            "Defense": "Combined Defense",
            "Technique": "Multi-layer protection",
            "Status": "Pending"
        }
    ]

    st.dataframe(
        defense_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### D1 versus D2")

    comparison_table = [

        {
            "Feature": "Detection method",
            "D1": "Regular expressions",
            "D2": "LLM classification"
        },

        {
            "Feature": "Semantic understanding",
            "D1": "Limited",
            "D2": "Model-dependent"
        },

        {
            "Feature": "Additional API requests",
            "D1": "No",
            "D2": "Yes"
        },

        {
            "Feature": "Detection explanation",
            "D1": "Matched rules",
            "D2": "Classification reason"
        },

        {
            "Feature": "Failure handling",
            "D1": "Local processing",
            "D2": "Configurable failure policy"
        },

        {
            "Feature": "Generalization performance",
            "D1": "Evaluation pending",
            "D2": "Evaluation pending"
        }
    ]

    st.dataframe(
        comparison_table,
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

    a1, a2, a3, a4 = st.columns(4)

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

    a4.metric(
        "Classifier Errors",
        st.session_state.classifier_error_count
    )

    history = st.session_state.request_history

    if history:

        st.divider()

        st.subheader("Session Defense Summary")

        summary = []

        for mode in DEFENSE_MODES:

            rows = [
                row for row in history
                if row["Defense"] == mode
            ]

            if not rows:
                continue

            latencies = [
                row["Latency (ms)"]
                for row in rows
                if row["Latency (ms)"] is not None
            ]

            token_values = [
                row["Combined Tokens"]
                for row in rows
                if row["Combined Tokens"] is not None
            ]

            summary.append({

                "Defense": mode,

                "Requests": len(rows),

                "Detected": (
                    sum(
                        row["Detected"] is True
                        for row in rows
                    )
                    if mode != "D0"
                    else "Not evaluated"
                ),

                "Blocked": sum(
                    row["Blocked"]
                    for row in rows
                ),

                "Classifier Errors": sum(
                    row["Classifier Errors"]
                    for row in rows
                ),

                "Average Latency (ms)": (
                    round(
                        sum(latencies) / len(latencies),
                        2
                    )
                    if latencies
                    else None
                ),

                "Average Combined Tokens": (
                    round(
                        sum(token_values) / len(token_values),
                        2
                    )
                    if token_values
                    else None
                )
            })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Average Session Latency")

        latency_data = [
            {
                "Defense": row["Defense"],
                "Average Latency (ms)":
                    row["Average Latency (ms)"]
            }
            for row in summary
            if row["Average Latency (ms)"] is not None
        ]

        if latency_data:

            st.bar_chart(
                latency_data,
                x="Defense",
                y="Average Latency (ms)"
            )

        st.subheader("Average Session Token Usage")

        token_data = [
            {
                "Defense": row["Defense"],
                "Average Combined Tokens":
                    row["Average Combined Tokens"]
            }
            for row in summary
            if row["Average Combined Tokens"] is not None
        ]

        if token_data:

            st.bar_chart(
                token_data,
                x="Defense",
                y="Average Combined Tokens"
            )

        st.caption(
            "Session averages may contain different "
            "questions and blocked requests. They are "
            "not controlled experimental comparisons."
        )

        with st.expander(
            "Session Request History"
        ):

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

    else:

        st.info(
            "Ask questions using D0, D1 and D2 "
            "to populate session analytics."
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
