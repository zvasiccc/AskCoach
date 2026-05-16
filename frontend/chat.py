import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="ChatWithAI", layout="centered")


def check_login(username, password):
    users = st.secrets.get("korisnici", {})
    user = users.get(username)
    if user and user["lozinka"] == password:
        result = dict(user)
        result["username"] = username
        return result
    return None


if "current_user" not in st.session_state:
    st.session_state["current_user"] = None

if st.session_state["current_user"] is None:
    st.title("Prijava")
    username_input = st.text_input("Korisničko ime")
    password_input = st.text_input("Lozinka", type="password")
    if st.button("Prijavi se"):
        user = check_login(username_input, password_input)
        if user:
            st.session_state["current_user"] = user
            st.rerun()
        else:
            st.error("Pogrešno korisničko ime ili lozinka.")
    st.stop()

current_user = st.session_state["current_user"]
role = current_user["uloga"]

st.title("Chat with AI")

with st.sidebar:
    st.caption(f"{current_user['ime']} — {role}")

    if role == "trener":
        all_users = st.secrets.get("korisnici", {})
        client_usernames = current_user.get("clients_ids", [])

        client_options = {
            all_users[k]["ime"]: k
            for k in client_usernames
            if k in all_users
        }

        if client_options:
            selected_client_name = st.selectbox("Izaberi klijenta:", list(client_options.keys()))
            selected_client_id = client_options[selected_client_name]
            selected_coach_id = current_user["username"]
        else:
            st.warning("Nema dodeljenih klijenata.")
            selected_client_id = None
            selected_coach_id = None
    else:
        selected_coach_id = current_user["coach_id"]
        selected_client_id = current_user["username"]
        st.caption(f"Tvoj trener: **{selected_coach_id}**")

    if st.button("Očisti konverzaciju"):
        st.session_state["messages"] = []
        st.rerun()

    if st.button("Odjavi se"):
        st.session_state["current_user"] = None
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
                    "coach_id": selected_coach_id,
                    "client_id": selected_client_id,
                    "question": user_input,
                    "history": st.session_state["messages"][:-1],
                    "role": role
                }
                response = requests.post(f"{API_URL}/ask", json=payload)

                if response.status_code == 200:
                    answer = response.json()["answer"]
                    context = response.json().get("context", [])
                else:
                    answer = "Greška pri komunikaciji sa serverom."
                    context = []
            except Exception as e:
                answer = f"Server nije dostupan: {e}"
                context = []

        st.write(answer)

        with st.expander("Debug"):
            st.write(f"**Collection:** {selected_coach_id} | **Client:** {selected_client_id}")
            for i, chunk in enumerate(context):
                st.text_area(f"Chunk {i+1}", value=chunk, height=80, disabled=True,
                             key=f"chunk_{len(st.session_state['messages'])}_{i}")

    st.session_state["messages"].append({"role": "assistant", "content": answer})