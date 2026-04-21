from __future__ import annotations

import streamlit as st

from src.rag_chatbot import answer_question, load_rag_pipeline


st.set_page_config(page_title="Rwanda Tax AI Assistant", page_icon="🇷🇼", layout="centered")


def _inject_css() -> None:
    """Small CSS tweaks for a clean ChatGPT-style layout."""
    st.markdown(
        """
<style>
/* Make the chat area feel more spacious */
.block-container { max-width: 900px; }

/* Right-align user messages a bit more */
div[data-testid="stChatMessage"][data-role="user"] {
  margin-left: 20%;
}

/* Left-align assistant messages a bit more */
div[data-testid="stChatMessage"][data-role="assistant"] {
  margin-right: 20%;
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def _get_rag() -> tuple:
    """
    Load the RAG pipeline once and reuse it across chat turns.

    IMPORTANT: We import and use the existing pipeline from src/rag_chatbot.py
    (we do NOT rebuild it in the UI).
    """
    return load_rag_pipeline()


def _init_session_state() -> None:
    """Initialize chat memory."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    _inject_css()
    _init_session_state()

    st.title("Rwanda Tax AI Assistant 🇷🇼")

    col_left, col_right = st.columns([1, 1], vertical_alignment="center")
    with col_left:
        st.caption("Ask questions and get answers grounded in your uploaded documents.")
    with col_right:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input at the bottom
    user_question = st.chat_input("Type your question…")
    if not user_question:
        return

    # Store + show the user's message
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate assistant response using existing RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                index, chunks, embedder, tokenizer, llm = _get_rag()
                answer = answer_question(
                    user_question,
                    index=index,
                    chunks=chunks,
                    embedder=embedder,
                    tokenizer=tokenizer,
                    llm=llm,
                )
            except Exception as exc:
                answer = f"Error: {exc}"

        st.markdown(answer)

    # Save assistant message to memory
    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()

