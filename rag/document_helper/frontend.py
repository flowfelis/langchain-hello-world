from typing import Any

import streamlit as st
from retieval import run_llm


def _format_sources(context_docs: list[Any]) -> list[str]:
    """Fetch urls for rendering citizions nicely"""
    return [
        str(meta.get("source") or "Unknown")
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(
    page_title="LangChain Documentation Assistant", layout="centered", page_icon="📚"
)
st.title("📚 LangChain Documentation Assistant")

with st.sidebar:
    st.subheader("Session")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop("messges", None)
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me anything from the LangChain documentation! I can provide answers and cite sources for you.",
            "sources": [],
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.markdown(f"- {source}")
prompt = st.chat_input("Ask a question about the LangChain documentation...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Let me check the documentation for you..."):
                result: dict[str, Any] = run_llm(prompt)
                answer = result.get("answer", "Sorry, I couldn't find an answer.")
                sources = _format_sources(result.get("context", []))
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            st.error(f"Sorry, something went wrong: {e}")
            st.exception(e)
