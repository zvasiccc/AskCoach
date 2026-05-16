import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="ChatWithAI - Upload", layout="centered")


def check_login(username, password):
    users = st.secrets.get("korisnici", {})
    user = users.get(username)
    if user and user["password"] == password:
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

# Samo trener moze da pristupi upload stranici
if role != "trener":
    st.error("Nemate pristup ovoj stranici.")
    st.stop()

st.title("Upravljanje bazom znanja")

with st.sidebar:
    st.caption(f"{current_user['full_name']} — {role}")
    if st.button("Odjavi se"):
        st.session_state["current_user"] = None
        st.rerun()

# ── UPLOAD ────────────────────────────────────────────────────────
st.header("Dodaj znanje")

all_users = st.secrets.get("korisnici", {})
client_usernames = current_user.get("clients_ids", [])

client_options = {
    all_users[k]["full_name"]: k
    for k in client_usernames
    if k in all_users
}

if not client_options:
    st.warning("Nemate dodeljenih klijenata.")
    st.stop()

selected_client_name = st.selectbox("Izaberi klijenta:", list(client_options.keys()))
selected_client_id = client_options[selected_client_name]
selected_coach_id = current_user["username"]

uploaded_file = st.file_uploader("Odaberite fajl", type=["txt", "pdf"])

if st.button("Učitaj u bazu"):
    if uploaded_file:
        with st.spinner("Obrađujem..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
            data = {
                "coach_id": selected_coach_id,
                "client_id": selected_client_id
            }
            try:
                response = requests.post(f"{API_URL}/upload", data=data, files=files)
                if response.status_code == 200:
                    st.success(response.json()["message"])
                    st.rerun()
                else:
                    st.error("Greška pri slanju na server.")
            except Exception as e:
                st.error(f"Server nije dostupan: {e}")
    else:
        st.warning("Odaberite fajl.")

st.divider()

# ── PREGLED BAZA ──────────────────────────────────────────────────
st.header("Postojeće baze znanja")

try:
    response = requests.get(f"{API_URL}/coaches")
    if response.status_code == 200:
        all_collections = response.json()["coaches"]

        # Trener vidi samo svoju kolekciju
        own_collections = [c for c in all_collections if c["id"] == selected_coach_id]

        if not own_collections:
            st.info("Nemate još uvek nijednu bazu znanja.")
        else:
            for collection in own_collections:
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.write(f"**{collection['id']}** — {collection['chunk_count']} chunkova")

                with col2:
                    if st.button("Pregled", key=f"preview_{collection['id']}"):
                        st.session_state["preview_collection"] = collection["id"]

                with col3:
                    if st.button("Obriši", key=f"delete_{collection['id']}"):
                        st.session_state["confirm_delete"] = collection["id"]

                if st.session_state.get("confirm_delete") == collection["id"]:
                    st.warning(f"Da li ste sigurni da želite da obrišete bazu **{collection['id']}**?")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Obriši", key=f"confirm_{collection['id']}"):
                            del_response = requests.delete(f"{API_URL}/coaches/{collection['id']}")
                            if del_response.status_code == 200:
                                st.success(f"Baza {collection['id']} obrisana.")
                                st.session_state.pop("confirm_delete", None)
                                st.rerun()
                    with c2:
                        if st.button("Otkaži", key=f"cancel_{collection['id']}"):
                            st.session_state.pop("confirm_delete", None)
                            st.rerun()

                if st.session_state.get("preview_collection") == collection["id"]:
                    preview_response = requests.get(f"{API_URL}/coaches/{collection['id']}/chunks")
                    if preview_response.status_code == 200:
                        chunks = preview_response.json()["chunks"]
                        st.markdown(f"**Chunkovi za {collection['id']}:**")
                        for i, chunk in enumerate(chunks[:10]):
                            st.text_area(f"Chunk {i+1}", value=chunk, height=80, disabled=True,
                                         key=f"chunk_{collection['id']}_{i}")
                        if len(chunks) > 10:
                            st.caption(f"... i još {len(chunks) - 10} chunkova")
                    if st.button("Zatvori pregled", key=f"close_{collection['id']}"):
                        st.session_state.pop("preview_collection", None)
                        st.rerun()
    else:
        st.error("Greška pri dohvatanju baza znanja.")
except Exception as e:
    st.error(f"Server nije dostupan: {e}")