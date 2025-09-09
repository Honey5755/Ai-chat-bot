import streamlit as st
import os, uuid, time
from chatbot import SupportChatbot

st.set_page_config(page_title="AI Support Chatbot", page_icon="🤖", layout='centered')

st.title("🤖 AI Customer Support (RAG + Memory)")
st.markdown("Upload support docs to `support_docs/` then run `python data_loader.py` to ingest.")


# initialize chatbot object once
if 'bot' not in st.session_state:
    try:
        st.session_state['bot'] = SupportChatbot()
    except Exception as e:
        st.error(f"Chatbot init failed: {e}")

# session id
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

# sidebar controls
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", options=["gpt-3.5-turbo", "gpt-4"], index=0)
    if st.button("Clear chat history"):
        st.session_state['session_id'] = str(uuid.uuid4())
        st.success("Cleared chat (new session started).")

# chat interface
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

def send_message(user_text):
    st.session_state['messages'].append(("user", user_text))
    with st.spinner('Thinking...'):
        answer, sources = st.session_state['bot'].ask(st.session_state['session_id'], user_text)
    st.session_state['messages'].append(("bot", answer))
    # show sources if present
    if sources:
        st.session_state['messages'].append(("system", "Sources: " + ", ".join(str(s.get('source','')) for s in sources)))

# input
user_input = st.chat_input("Ask a question about the product / service...")
if user_input:
    send_message(user_input)

# display history
for role, msg in st.session_state['messages']:
    if role == 'user':
        st.chat_message('user').markdown(msg)
    elif role == 'bot':
        st.chat_message('assistant').markdown(msg)
    else:
        st.info(msg)

st.markdown('---')
st.markdown('**Developer notes:** Run `python data_loader.py` after adding docs to `support_docs/`.')
