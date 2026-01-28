import os
import pgserver
import cocoindex
from cocoindex import Settings, DatabaseConnectionSpec
import time
import sys

import logging
logging.basicConfig(level=logging.INFO)  # Start with INFO, DEBUG might be too noisy with PG

def main():
    # 1. Start pgserver (REQUIRED for CoCo's Internal State)
    PG_DATA_DIR = "./my_coco_data"
    print(f"🚀 Starting pgserver (Internal State) in {PG_DATA_DIR}...")
    try:
        db = pgserver.get_server(PG_DATA_DIR)
        db.psql("CREATE EXTENSION IF NOT EXISTS vector;")
    except Exception as e:
        print(f"❌ Failed to start pgserver: {e}")
        sys.exit(1)

    # 2. Patch URI
    raw_uri = db.get_uri()
    if "@/" in raw_uri and "?host=" in raw_uri:
        clean_uri = raw_uri.replace("@/", "@localhost/")
    else:
        clean_uri = raw_uri
        
    os.environ["COCO_DB_URL"] = clean_uri
    print(f"✨ CoCo Internal DB: {clean_uri}")

    # 3. Configure CocoIndex
    coco_settings = Settings(
        app_namespace="my_app",
        database=DatabaseConnectionSpec(url=clean_uri)
    )

    try:
        cocoindex.init(coco_settings)
        print("✅ CocoIndex initialized.")
        
        # 4. Import and Register Flow
        print("🌊 Setting up Ingestion Flow...")
        from coco_app.flow import ingestion_flow
        from cocoindex import FlowLiveUpdater, ServerSettings, open_flow
        
        # 1. Open Flow Handle (Method B with Decorator)
        # We pass the function itself.
        flow_handle = open_flow("IngestionFlow", ingestion_flow)
        
        # 2. Setup metadata
        flow_handle.setup(report_to_stdout=True)
        
        print("✅ Flow setup complete.")
        
        # 3. Start UI Server (Background/Non-blocking)
        cocoindex.start_server(ServerSettings(cors_origins=[]))
        
        print("✅ Server started. Starting Live Updater...")
        
        # 4. Running Loop
        with FlowLiveUpdater(flow_handle) as updater:
            print("🚀 FlowLiveUpdater active! Watching for file changes...")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
