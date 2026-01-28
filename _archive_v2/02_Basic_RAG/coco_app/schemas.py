from cocoindex import Table, Vector, String, Json

# The Unified Schema for all RAG data
# This table will store chunks from PDFs, CSVs, etc.
class UnifiedTable(Table):
    content: String
    embedding: Vector(768)  # nomic-embed-text is 768d
    source_id: String
    metadata: Json
