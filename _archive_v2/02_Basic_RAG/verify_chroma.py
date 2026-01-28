import chromadb

def verify():
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        coll = client.get_collection("coco_collection")
        count = coll.count()
        print(f"✅ Collection found. Total documents: {count}")
        
        if count > 0:
            print("🔍 Sample Document:")
            print(coll.peek(limit=1))
    except Exception as e:
        print(f"❌ Verification Failed: {e}")

if __name__ == "__main__":
    verify()
