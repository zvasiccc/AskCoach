from db.chroma import ChromaDBManager
from ingest.chunker import DocumentChunker

db = ChromaDBManager()
chunker = DocumentChunker()

test_text = "Kreatin se pije 5g dnevno. Najbolje ga je uzimati nakon treninga uz obrok sa hidratima."
chunks = chunker.chunk_text(test_text)

db.add_to_collection(
    coach_id="1",
    ids=["id1"],
    documents=chunks,
    metadatas=[{"source": "test_video_1"}]
)

print("Uspešno kreirana kolekcija i ubačen dokument!")