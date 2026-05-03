import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AskCoach", layout="centered")
st.title("🏋️‍♂️ AskCoach")

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Podešavanja")
    
    # Dohvati listu trenera
    try:
        response = requests.get(f"{API_URL}/coaches")
        coaches = response.json().get("coaches", [])
        coach_options = [c["id"] for c in coaches]
    except Exception:
        coach_options = []
        st.error("Server nije dostupan.")

    if coach_options:
        selected_coach = st.selectbox("Izaberi trenera:", coach_options)
    else:
        selected_coach = st.text_input("ID Trenera:", value="trener_zeljko")

    if st.button("🗑️ Očisti razgovor"):
        st.session_state["messages"] = []
        st.rerun()

# ─── SESSION STATE ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ─── PRIKAZ ISTORIJE ──────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ─── INPUT ────────────────────────────────────────────────────
if user_input := st.chat_input("Postavi pitanje treneru..."):
    # Prikaži korisnikovu poruku
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Pošalji na API
    with st.chat_message("assistant"):
        with st.spinner("Razmišljam..."):
            try:
                payload = {
                    "coach_id": selected_coach,
                    "pitanje": user_input,
                    "history": st.session_state["messages"][:-1]
                }
                response = requests.post(f"{API_URL}/ask", json=payload)
                
                if response.status_code == 200:
                    odgovor = response.json()["odgovor"]
                    context = response.json().get("context", [])
                else:
                    odgovor = "Greška pri komunikaciji sa serverom."
                    context = []
            except Exception as e:
                odgovor = f"Server nije dostupan: {e}"
                context = []

        st.write(odgovor)
        
        with st.expander("🔍 Debug info"):
            st.write(f"**Trener:** {selected_coach}")
            for i, chunk in enumerate(context):
                st.text_area(f"Chunk {i+1}", value=chunk, height=80, disabled=True, key=f"chunk_{len(st.session_state['messages'])}_{i}")

    # Sačuvaj odgovor u history
    st.session_state["messages"].append({"role": "assistant", "content": odgovor})