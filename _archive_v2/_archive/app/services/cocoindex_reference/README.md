# CocoIndex Reference Implementation

This folder contains archived CocoIndex code from the CoCo RAG project for future reference.

## Overview

CocoIndex is a document indexing library that provides:
- Live file watching with automatic updates
- Integration with PostgreSQL for state management
- Flow-based document processing pipelines

## Files

- `flow.py` - CocoIndex flow definition
- `cocoindex_logic.md` - Detailed implementation notes (below)
- Original files preserved for reference

## Dependencies Required

```
cocoindex>=0.3.25
pgserver>=0.1.4
```

## Quick Reference

### Starting CocoIndex Server

```python
import pgserver
import cocoindex
from cocoindex import Settings, DatabaseConnectionSpec

# 1. Start pgserver (embedded PostgreSQL)
db = pgserver.get_server("./my_coco_data")
db.psql("CREATE EXTENSION IF NOT EXISTS vector;")

# 2. Get connection URI
raw_uri = db.get_uri()
clean_uri = raw_uri.replace("@/", "@localhost/")

# 3. Initialize CocoIndex
coco_settings = Settings(
    app_namespace="my_app",
    database=DatabaseConnectionSpec(url=clean_uri)
)
cocoindex.init(coco_settings)
```

### Defining a Flow

```python
from cocoindex import FlowBuilder, DataScope, op
from cocoindex.sources import LocalFile
from datetime import timedelta

@op.function()
def process_file(filename: str) -> int:
    """Process a single file."""
    # Your processing logic here
    return chunk_count

def ingestion_flow(flow_builder: FlowBuilder, data_scope: DataScope):
    # Add file source with auto-refresh
    data_scope["files"] = flow_builder.add_source(
        LocalFile(path="./data/pdfs"),
        refresh_interval=timedelta(seconds=5)
    )
    
    # Process each file
    with data_scope["files"].row() as file:
        file["result"] = file["filename"].transform(process_file)
```

### Running with Live Updates

```python
from cocoindex import FlowLiveUpdater, ServerSettings, open_flow

# Open flow handle
flow_handle = open_flow("IngestionFlow", ingestion_flow)
flow_handle.setup(report_to_stdout=True)

# Start UI server (optional)
cocoindex.start_server(ServerSettings(cors_origins=[]))

# Run with live updates
with FlowLiveUpdater(flow_handle) as updater:
    print("Watching for file changes...")
    while True:
        time.sleep(1)
```
