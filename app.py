
import streamlit as st

from statistics import mean

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
    "D0": "Baseline RAG",
    "D1": "Rule-Based Detection",
    "D2": "LLM-Based Detection",
    "D3": "Context Isolation",
    "D4": "Output Validation",
    "D5": "Combined Defense"
}

DEFENSE_DESCRIPTIONS = {
    "D0": (
        "Baseline RAG without additional "
        "experimental security defenses."
    ),
    "D1": (
        "Rule-based scanning of user input "
        "and retrieved documents."
    ),
    "D2": (
        "LLM-based classification of user input "
        "and retrieved documents."
    ),
    "D3": (
        "Structured context and explicit separation "
        "of trusted instructions from untrusted data."
    ),
    "D4": (
        "Post-generation validation against "
        "configured output security policies."
    ),
    "D5": (
        "Combined rule-based detection, LLM "
        "classification, context isolation "
        "and output validation."
    )
}

LAYER_NAMES = {
    "D1": "Rule-Based Detection",
    "D2": "LLM-Based Detection",
    "D3": "Context Isolation",
    "D4": "Output Validation"
}

DETECTION_MODES = {"D1", "D2", "D4", "D5"}


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

def session_defaults():

    return {
        "messages": [],
        "last_result": None,
        "request_history": [],
        "request_count": 0,
        "detected_count": 0,
        "blocked_count": 0,
        "classifier_error_count": 0,
        "request_error_count": 0
    }


for key, value in session_defaults().items():

    if key not in st.session_state:

        st.session_state[key] = value


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def reset_session():

    for key, value in session_defaults().items():

        st.session_state[key] = value


def format_tokens(value):

    if value is None:
        return "Unavailable"

    return f"{value:,}"


def format_latency(value):

    if value is None:
        return "Unavailable"

    return f"{value:.2f} ms"


def detection_status(result):

    if result.get("defense_mode") not in DETECTION_MODES:

        return "Not evaluated"

    if result.get("attack_detected"):

        return "Detected"

    if result.get("classifier_errors", 0):

        return "Incomplete"

    executed = set(result.get("executed_layers", []))

    if not executed.intersection({"D1", "D2", "D4"}):

        return "Not evaluated"

    return "Not detected"


def record_result(result):

    st.session_state.last_result = result

    st.session_state.request_count += 1

    st.session_state.detected_count += int(
        bool(result.get("attack_detected"))
    )

    st.session_state.blocked_count += int(
        bool(result.get("blocked"))
    )

    st.session_state.classifier_error_count += (
        result.get("classifier_errors", 0)
    )

    st.session_state.request_history.append({
        "Defense": result["defense_mode"],
        "Model": result["model"],
        "Detection": detection_status(result),
        "Blocked": result.get("blocked", False),
        "Blocked By": result.get("blocked_by"),
        "Blocked Stage": result.get("blocked_stage"),
        "Latency (ms)": result.get("total_latency_ms"),
        "Defense Latency (ms)":
            result.get("defense_latency_ms"),
        "Generation Tokens": result.get("total_tokens"),
        "Defense Tokens":
            result.get("defense_total_tokens"),
        "Combined Tokens":
            result.get("combined_total_tokens"),
        "Classifier Requests":
            result.get("defense_api_calls", 0),
        "Classifier Errors":
            result.get("classifier_errors", 0),
        "Isolation":
            result.get("isolation_applied", False),
        "Output Validation":
            result.get("output_validation_status"),
        "Status": result.get("execution_status")
    })


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("🛡️ RAG Security Lab")

    st.caption("Prompt Injection Defense Evaluation")

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
        list(DEFENSE_MODES),
        format_func=lambda mode: (
            f"{mode} — {DEFENSE_MODES[mode]}"
        )
    )

    if defense_mode == "D0":

        st.warning(
            "Experimental defenses disabled."
        )

    else:

        st.success(
            DEFENSE_DESCRIPTIONS[defense_mode]
        )

    if defense_mode in {"D2", "D5"}:

        st.caption(
            "This configuration may require "
            "additional classifier API requests."
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

    st.caption("Generation API: Groq configuration")

    st.divider()

    st.subheader("Session Statistics")

    st.metric(
        "Processed Requests",
        st.session_state.request_count
    )

    st.metric(
        "Detected Requests",
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

    st.metric(
        "Request Errors",
        st.session_state.request_error_count
    )

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
    "Evaluating the Generalization of Prompt-Injection "
    "Defenses in Retrieval-Augmented Generation Systems"
)


# ==================================================
# RESPONSE DIAGNOSTICS
# ==================================================

def render_diagnostics(result):

    st.divider()

    st.subheader("Response Diagnostics")

    st.caption(
        f"Model: {result['model']} | "
        f"Defense: {result['defense_mode']} | "
        f"Status: {result['execution_status']}"
    )

    if result["defense_mode"] != defense_mode:

        st.info(
            "These diagnostics belong to "
            "the previous request."
        )

    # ----------------------------------------------
    # LATENCY
    # ----------------------------------------------

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
            result.get("defense_latency_ms")
        )
    )

    c4.metric(
        "Total Latency",
        format_latency(
            result.get("total_latency_ms")
        )
    )

    # ----------------------------------------------
    # TOKEN USAGE
    # ----------------------------------------------

    t1, t2, t3, t4 = st.columns(4)

    t1.metric(
        "Generation Input",
        format_tokens(result.get("input_tokens"))
    )

    t2.metric(
        "Generation Output",
        format_tokens(result.get("output_tokens"))
    )

    t3.metric(
        "Defense Tokens",
        format_tokens(
            result.get("defense_total_tokens")
        )
    )

    t4.metric(
        "Combined Tokens",
        format_tokens(
            result.get("combined_total_tokens")
        )
    )

    # ----------------------------------------------
    # SECURITY DIAGNOSTICS
    # ----------------------------------------------

    st.divider()

    st.subheader("Security Diagnostics")

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Detection",
        detection_status(result)
    )

    s2.metric(
        "Blocked",
        "Yes" if result.get("blocked") else "No"
    )

    s3.metric(
        "Excluded Chunks",
        len(result.get("flagged_chunks", []))
    )

    s4.metric(
        "Generation Called",
        "Yes" if result.get("llm_called") else "No"
    )

    if result.get("execution_status") == "defense_error":

        st.error(
            "Security classification was unavailable."
        )

    elif result.get("blocked"):

        st.error(
            f"Blocked by {result.get('blocked_by')} "
            f"at {result.get('blocked_stage')}."
        )

    elif result.get("classifier_errors"):

        st.warning(
            "One or more classifier requests failed."
        )

    elif result.get("attack_detected"):

        st.warning(
            "Suspicious content was detected "
            "by the active defense."
        )

    elif result["defense_mode"] == "D0":

        st.info(
            "No experimental security defense active."
        )

    elif result["defense_mode"] == "D3":

        st.info(
            "Context isolation does not perform "
            "attack classification."
        )

    else:

        st.success(
            "No violation detected by executed checks. "
            "This does not guarantee a secure response."
        )

    # ----------------------------------------------
    # DEFENSE LAYER EXECUTION
    # ----------------------------------------------

    st.subheader("Defense Layer Execution")

    active = result.get("active_layers", [])

    executed = result.get("executed_layers", [])

    layer_columns = st.columns(4)

    for index, layer in enumerate(LAYER_NAMES):

        with layer_columns[index]:

            st.markdown(
                f"**{layer} — {LAYER_NAMES[layer]}**"
            )

            if layer in executed:

                st.success("Executed")

            elif layer in active:

                st.warning("Not reached")

            else:

                st.caption("Disabled")

    # ----------------------------------------------
    # D1 RULE DIAGNOSTICS
    # ----------------------------------------------

    matched_rules = result.get("matched_rules", [])

    if matched_rules:

        st.subheader("Matched Security Rules")

        for rule in matched_rules:

            with st.expander(
                str(rule.get("rule_id", "Rule"))
            ):

                st.json(rule)

    # ----------------------------------------------
    # D2 CLASSIFIER DIAGNOSTICS
    # ----------------------------------------------

    if "D2" in active:

        st.divider()

        st.subheader("D2 — LLM Classifier Diagnostics")

        scans = result.get("classifier_results", [])

        st.caption(
            "Classifier model: "
            + str(result.get("classifier_model"))
        )

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Classifier Requests",
            result.get("defense_api_calls", 0)
        )

        d2.metric(
            "Benign",
            sum(
                scan.get("label") == "BENIGN"
                for scan in scans
            )
        )

        d3.metric(
            "Adversarial",
            sum(
                scan.get("label") == "ADVERSARIAL"
                for scan in scans
            )
        )

        d4.metric(
            "Unknown",
            sum(
                scan.get("label") == "UNKNOWN"
                for scan in scans
            )
        )

        if not scans:

            st.info(
                "Classifier stage was not reached."
            )

        for index, scan in enumerate(scans, 1):

            with st.expander(
                f"Classification {index} — "
                f"{scan.get('channel')} — "
                f"{scan.get('label')}"
            ):

                st.write(
                    "Category:",
                    scan.get("category")
                )

                st.write(
                    "Reason:",
                    scan.get("reason")
                )

                st.write(
                    "Action:",
                    scan.get("action")
                )

                st.write(
                    "Latency:",
                    format_latency(
                        scan.get("detection_latency_ms")
                    )
                )

                st.json(scan)

    # ----------------------------------------------
    # D3 CONTEXT ISOLATION DIAGNOSTICS
    # ----------------------------------------------

    if "D3" in active:

        st.divider()

        st.subheader("D3 — Context Isolation")

        i1, i2, i3 = st.columns(3)

        i1.metric(
            "Isolation Applied",
            "Yes" if result.get(
                "isolation_applied"
            ) else "No"
        )

        i2.metric(
            "Isolated Chunks",
            (
                len(result.get("context_used", []))
                if result.get("isolation_applied")
                else 0
            )
        )

        i3.metric(
            "Isolation Latency",
            format_latency(
                result.get("isolation_latency_ms")
            )
        )

        if result.get("isolation_applied"):

            st.success(
                "Trusted instructions and untrusted "
                "retrieved content were separated."
            )

            st.caption(
                "Logical prompt isolation is not "
                "a guaranteed security boundary."
            )

        else:

            st.info(
                "Context isolation stage was not reached."
            )

    # ----------------------------------------------
    # D4 OUTPUT VALIDATION DIAGNOSTICS
    # ----------------------------------------------

    if "D4" in active:

        st.divider()

        st.subheader("D4 — Output Validation")

        status = result.get(
            "output_validation_status",
            "not_evaluated"
        )

        violations = result.get(
            "output_violations", []
        )

        v1, v2, v3 = st.columns(3)

        v1.metric(
            "Validation Status",
            status.replace("_", " ").title()
        )

        v2.metric(
            "Policy Violations",
            len(violations)
        )

        v3.metric(
            "Validation Latency",
            format_latency(
                result.get("validation_latency_ms")
            )
        )

        if status == "passed":

            st.success(
                "Response passed configured "
                "output security checks."
            )

        elif status == "blocked":

            st.error(
                "Generated response was suppressed."
            )

        elif status == "empty_response":

            st.warning(
                "The model returned an empty response."
            )

        else:

            st.info(
                "Output validation stage was not reached."
            )

        for violation in violations:

            with st.expander(
                violation.get("rule_id", "Violation")
            ):

                st.json(violation)

    # ----------------------------------------------
    # KNOWLEDGE RETRIEVAL
    # ----------------------------------------------

    st.divider()

    st.subheader("Knowledge Retrieval")

    retrieved = result.get("retrieved_chunks", [])

    used = result.get("context_used", [])

    flagged = result.get("flagged_chunks", [])

    used_ids = {
        chunk["chunk_id"] for chunk in used
    }

    r1, r2, r3 = st.columns(3)

    r1.metric("Retrieved Chunks", len(retrieved))

    r2.metric("Context Used", len(used))

    r3.metric("Excluded Chunks", len(flagged))

    with st.expander("Retrieved Source Documents"):

        if not retrieved:

            st.info("No documents retrieved.")

        for chunk in retrieved:

            st.markdown(
                f"### Rank {chunk['rank']} — "
                f"{chunk['document_id']}"
            )

            st.caption(
                f"Domain: {chunk['domain']} | "
                f"Similarity: {chunk['similarity']:.4f}"
            )

            if chunk["chunk_id"] in used_ids:

                st.success(
                    "Included in generation context"
                )

            else:

                st.warning(
                    "Not included in generation context"
                )

            st.write(chunk["text"])

            st.caption(
                f"Source: {chunk['source']}"
            )

            st.divider()

    if flagged:

        with st.expander("Excluded Context Chunks"):

            for chunk in flagged:

                st.write(chunk["document_id"])

                st.write(chunk["text"])

                st.divider()

    with st.expander("Defense Execution Details"):

        st.json(
            result.get("defense_details", {})
        )

    with st.expander("Raw Response Metadata"):

        st.json(result)


# ==================================================
# AI CHATBOT PAGE
# ==================================================

if page == "AI Chatbot":

    st.subheader("Institutional Knowledge Assistant")

    st.info(
        f"{defense_mode} — "
        f"{DEFENSE_DESCRIPTIONS[defense_mode]}"
    )

    result = st.session_state.last_result

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Selected Defense",
        defense_mode
    )

    c2.metric(
        "Processed Requests",
        st.session_state.request_count
    )

    c3.metric(
        "Last Latency",
        format_latency(
            result.get("total_latency_ms")
        ) if result else "—"
    )

    c4.metric(
        "Last Combined Tokens",
        format_tokens(
            result.get("combined_total_tokens")
        ) if result else "—"
    )

    st.divider()

    if not st.session_state.messages:

        with st.chat_message("assistant"):

            st.markdown(
                "Welcome to RAG Security Lab.\n\n"
                "Ask questions about institutional "
                "policies, attendance, academics, "
                "examinations, placements, library "
                "services, hostel rules and admissions."
            )

            st.caption(
                "All six defense configurations "
                "are available in the sidebar. "
                "Questions are currently processed "
                "independently."
            )

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if message.get("defense_mode"):

                st.caption(
                    "Defense: " + message["defense_mode"]
                )

            if message.get("blocked"):

                st.warning(
                    "Blocked by security defense."
                )

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

        try:

            with st.spinner(
                "Retrieving documents and "
                "executing selected defenses..."
            ):

                result = load_pipeline().ask(
                    question=question,
                    top_k=top_k,
                    defense_mode=defense_mode
                )

            answer = result.get("answer") or (
                "The model returned an empty response."
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "defense_mode": defense_mode,
                "blocked": result.get("blocked", False)
            })

            record_result(result)

        except Exception as error:

            st.session_state.request_error_count += 1

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"Request failed: "
                    f"{type(error).__name__}. "
                    "Check the API configuration "
                    "and backend."
                ),
                "defense_mode": defense_mode
            })

        st.rerun()

    if st.session_state.last_result:

        render_diagnostics(
            st.session_state.last_result
        )


# ==================================================
# ATTACK SIMULATOR PAGE
# ==================================================

elif page == "Attack Simulator":

    st.subheader("Attack Simulator")

    st.info(
        "The defense backend is available. "
        "Controlled benchmark execution will "
        "be implemented in the next phase."
    )

    st.markdown("### Planned Attack Families")

    families = [
        {
            "Attack Family": "Direct Prompt Injection",
            "Status": "Controlled samples available"
        },
        {
            "Attack Family": "Instruction Manipulation",
            "Status": "Scenario preparation"
        },
        {
            "Attack Family": "Obfuscation",
            "Status": "Scenario preparation"
        },
        {
            "Attack Family": "Indirect Prompt Injection",
            "Status": "Dataset integration pending"
        },
        {
            "Attack Family": "Multi-stage Prompt Injection",
            "Status": "Conversation integration pending"
        }
    ]

    st.dataframe(
        families,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### Available Defenses")

    for mode, name in DEFENSE_MODES.items():

        st.write(f"{mode} — {name}")


# ==================================================
# DEFENSE COMPARISON PAGE
# ==================================================

elif page == "Defense Comparison":

    st.subheader("Defense Comparison")

    st.caption(
        "Implementation capabilities, "
        "not measured security performance."
    )

    rows = []

    for mode, name in DEFENSE_MODES.items():

        rows.append({
            "Mode": mode,
            "Defense": name,
            "Rule-Based":
                mode in {"D1", "D5"},
            "LLM Detection":
                mode in {"D2", "D5"},
            "Context Isolation":
                mode in {"D3", "D5"},
            "Output Validation":
                mode in {"D4", "D5"}
        })

    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    st.markdown("### Defense Mechanisms")

    for mode, name in DEFENSE_MODES.items():

        with st.expander(f"{mode} — {name}"):

            st.write(
                DEFENSE_DESCRIPTIONS[mode]
            )

    st.markdown("### Pairwise Ablation Configurations")

    st.write(
        "D1+D2, D1+D3, D1+D4, "
        "D2+D3, D2+D4, D3+D4"
    )

    st.info(
        "Pairwise configurations are available "
        "in the backend. Formal side-by-side "
        "benchmark comparison is pending."
    )


# ==================================================
# ANALYTICS PAGE
# ==================================================

elif page == "Analytics":

    st.subheader("Research Analytics")

    st.caption(
        "Live session statistics, "
        "not formal benchmark results."
    )

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Processed Requests",
        st.session_state.request_count
    )

    a2.metric(
        "Detected Requests",
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

            defense_latencies = [
                row["Defense Latency (ms)"]
                for row in rows
                if row["Defense Latency (ms)"]
                is not None
            ]

            tokens = [
                row["Combined Tokens"]
                for row in rows
                if row["Combined Tokens"] is not None
            ]

            summary.append({
                "Defense": mode,
                "Requests": len(rows),
                "Detected": (
                    sum(
                        row["Detection"] == "Detected"
                        for row in rows
                    )
                    if mode in DETECTION_MODES
                    else None
                ),
                "Blocked": sum(
                    row["Blocked"] for row in rows
                ),
                "Classifier Errors": sum(
                    row["Classifier Errors"]
                    for row in rows
                ),
                "Average Latency (ms)": (
                    round(mean(latencies), 2)
                    if latencies else None
                ),
                "Average Defense Latency (ms)": (
                    round(mean(defense_latencies), 2)
                    if defense_latencies else None
                ),
                "Average Combined Tokens": (
                    round(mean(tokens), 2)
                    if tokens else None
                )
            })

        st.subheader("Session Defense Summary")

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Average Response Latency")

        st.bar_chart(
            summary,
            x="Defense",
            y="Average Latency (ms)"
        )

        st.subheader("Average Defense Overhead")

        st.bar_chart(
            summary,
            x="Defense",
            y="Average Defense Latency (ms)"
        )

        st.subheader("Average Combined Token Usage")

        token_rows = [
            row for row in summary
            if row["Average Combined Tokens"] is not None
        ]

        if token_rows:

            st.bar_chart(
                token_rows,
                x="Defense",
                y="Average Combined Tokens"
            )

        with st.expander("Session Request History"):

            st.dataframe(
                history,
                hide_index=True,
                use_container_width=True
            )

        st.caption(
            "Session averages may contain different "
            "questions, blocked requests and "
            "incomplete classifications. They are "
            "not controlled research comparisons."
        )

    else:

        st.info(
            "Submit questions using different "
            "defenses to populate analytics."
        )

    st.divider()

    st.markdown("### Planned Formal Research Metrics")

    st.markdown(
        """
        - Attack success rate
        - Detection precision and recall
        - False positive rate and F1 score
        - Answer utility and grounding
        - Seen versus unseen generalization
        - Defense latency and token consumption
        - API cost
        - Ablation and statistical significance
        """
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
            placeholder="Document ID, domain or content"
        ).strip().casefold()

        filtered = []

        for document in documents:

            if (
                selected_domain != "All"
                and document.domain != selected_domain
            ):

                continue

            searchable = (
                document.document_id
                + " "
                + document.domain
                + " "
                + document.text
            ).casefold()

            if search_query not in searchable:

                continue

            filtered.append(document)

        st.caption(
            f"Showing {len(filtered)} "
            f"of {len(documents)} documents"
        )

        if not filtered:

            st.warning(
                "No matching documents found."
            )

        for document in filtered:

            with st.expander(
                f"{document.document_id} — "
                f"{document.domain.title()}"
            ):

                st.caption(
                    f"Domain: {document.domain}"
                )

                st.write(document.text)

                st.caption(
                    f"Source: {document.source}"
                )

    except Exception as error:

        st.error(
            "Unable to load documents: "
            + type(error).__name__
        )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "RAG Security Lab | Institutional RAG "
    "Security Research | B.Tech Major Project"
)
