import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="ChatWithAI", layout="centered")
st.title("Chat with AI")

with st.sidebar:

    try:
        coaches_list = requests.get(f"{API_URL}/coaches")
        coaches = coaches_list.json().get("coaches", [])
        coach_options = [c["id"] for c in coaches]
    except Exception:
        coach_options = []
        st.error("Server nije dostupan.")

    if coach_options:
        selected_coach = st.selectbox(" ", coach_options)
    else:
        selected_coach = st.text_input("ID Baze znanja:", value="trener_zeljko")

    if st.button("Ocisti konverzaciju"):
        st.session_state["messages"] = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Postavi pitanje..."):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Razmišljam..."):
            try:
                payload = {
                    "coach_id": selected_coach,
                    "pitanje": user_input,
                    "history": st.session_state["messages"][:-1]
                }
                coaches_list = requests.post(f"{API_URL}/ask", json=payload)
                
                if coaches_list.status_code == 200:
                    answer = coaches_list.json()["answer"]
                    context = coaches_list.json().get("context", [])
                else:
                    answer = "Greska pri komunikaciji sa serverom."
                    context = []
            except Exception as e:
                answer = f"Server nije dostupan: {e}"
                context = []

        st.write(answer)
        
        with st.expander("Debug"):
            st.write(f"**Baza znanja:** {selected_coach}")
            for i, chunk in enumerate(context):
                st.text_area(f"Chunk {i+1}", value=chunk, height=80, disabled=True, key=f"chunk_{len(st.session_state['messages'])}_{i}")

    #cuva answer u history
    st.session_state["messages"].append({"role": "assistant", "content": answer})