import streamlit as st
import requests

st.set_page_config(page_title="AskCoach - Admin Panel", layout="centered")

st.title("🏋️‍♂️ Upravljanje bazom znanja trenera")
st.write("Dodajte nove informacije u bazu znanja specifičnog trenera.")

# 1. Odabir trenera (Izolacija baze)
coach_id = st.text_input("ID Trenera (npr. trener_zeljko, trener_vukadin)", value="trener_zeljko")

# 2. Upload fajla
uploaded_file = st.file_uploader("Odaberite .txt fajl sa znanjem", type=["txt"])

if st.button("Učitaj u bazu"):
    if uploaded_file is not None and coach_id:
        with st.spinner("Obrađujem i indeksiram..."):
            # Slanje na backend
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
            data = {"coach_id": coach_id}
            
            try:
                response = requests.post("http://localhost:8000/upload", data=data, files=files)
                if response.status_code == 200:
                    st.success(response.json()["message"])
                else:
                    st.error("Greška pri slanju na server.")
            except Exception as e:
                st.error(f"Server nije dostupan: {e}")
    else:
        st.warning("Molimo unesite ID trenera i odaberite fajl.")

st.sidebar.info("Ovaj panel služi za Advanced RAG administraciju. Podaci se automatski pretvaraju u vektore i šalju u ChromaDB.")