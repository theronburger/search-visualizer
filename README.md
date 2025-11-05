# Search Results Visualizer

Interactive analysis of vector search performance across different query strategies and nonprofits.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create a `config.py` file:

```python
MONGODB_URI = "your_mongodb_connection_string"
DATABASE_NAME = "your_database_name"
COLLECTION_NAME = "your_collection_name"
```

## Run

```bash
marimo edit search_analysis.py
```

## Visualizations

- Aggregate metrics by query_type
- Heatmap: query_type × nonprofit with avg rank/reciprocal_rank
- Success rate analysis (top_1, top_5, top_10, etc.)
- Distribution plots of ranks
- Interactive filtering and drill-down
