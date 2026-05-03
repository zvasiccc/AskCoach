import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AskCoach - Admin Panel", layout="centered")
st.title("🏋️‍♂️ Upravljanje bazom znanja trenera")

# ─── UPLOAD SEKCIJA ───────────────────────────────────────────
st.header("Dodaj znanje treneru")

coach_id = st.text_input("ID Trenera (npr. trener_zeljko)", value="trener_zeljko")
uploaded_file = st.file_uploader("Odaberite .txt fajl", type=["txt"])

if st.button("Učitaj u bazu"):
    if uploaded_file and coach_id:
        with st.spinner("Obrađujem i indeksiram..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
            data = {"coach_id": coach_id}
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
        st.warning("Unesite ID trenera i odaberite fajl.")

st.divider()

# ─── PREGLED BAZA ─────────────────────────────────────────────
st.header("Baze trenera")

try:
    response = requests.get(f"{API_URL}/coaches")
    if response.status_code == 200:
        coaches = response.json()["coaches"]

        if not coaches:
            st.info("Nema unesenih trenera u bazi.")
        else:
            for coach in coaches:
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.write(f"👤 **{coach['id']}** — {coach['chunk_count']} chunkova")

                with col2:
                    if st.button("🔍 Pregled", key=f"preview_{coach['id']}"):
                        st.session_state["preview_coach"] = coach["id"]

                with col3:
                    if st.button("🗑️ Obriši", key=f"delete_{coach['id']}"):
                        st.session_state["confirm_delete"] = coach["id"]

                # Potvrda brisanja
                if st.session_state.get("confirm_delete") == coach["id"]:
                    st.warning(f"Sigurno želiš da obrišeš bazu za **{coach['id']}**?")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Da, obriši", key=f"confirm_{coach['id']}"):
                            del_response = requests.delete(f"{API_URL}/coaches/{coach['id']}")
                            if del_response.status_code == 200:
                                st.success(f"Baza za {coach['id']} obrisana.")
                                st.session_state.pop("confirm_delete", None)
                                st.rerun()
                    with c2:
                        if st.button("❌ Otkaži", key=f"cancel_{coach['id']}"):
                            st.session_state.pop("confirm_delete", None)
                            st.rerun()

                # Pregled chunkova
                if st.session_state.get("preview_coach") == coach["id"]:
                    preview_response = requests.get(f"{API_URL}/coaches/{coach['id']}/chunks")
                    if preview_response.status_code == 200:
                        chunks = preview_response.json()["chunks"]
                        st.markdown(f"**Chunkovi za {coach['id']}:**")
                        for i, chunk in enumerate(chunks[:10]):
                            st.text_area(f"Chunk {i+1}", value=chunk, height=80, disabled=True, key=f"chunk_{coach['id']}_{i}")
                        if len(chunks) > 10:
                            st.caption(f"... i još {len(chunks) - 10} chunkova")
                    if st.button("Zatvori pregled", key=f"close_{coach['id']}"):
                        st.session_state.pop("preview_coach", None)
                        st.rerun()
    else:
        st.error("Greška pri dohvatanju trenera.")
except Exception as e:
    st.error(f"Server nije dostupan: {e}")

st.sidebar.info("AskCoach Admin Panel — upravljanje vektorskim bazama trenera.")