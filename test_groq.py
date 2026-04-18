import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def test_groq():
    # Inicijalizacija LLM-a
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    print("🤖 Šaljem upit Groq-u...")
    pitanje = "Zdravo! Ti si moj novi fitnes asistent. Kratko reci koji je tvoj cilj?"
    
    try:
        response = llm.invoke(pitanje)
        print("-" * 30)
        print(f"ODGOVOR: {response.content}")
        print("-" * 30)
    except Exception as e:
        print(f"❌ Greška kod Groq-a: {e}")

if __name__ == "__main__":
    test_groq()