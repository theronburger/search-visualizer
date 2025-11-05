import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pymongo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    from datetime import datetime
    from pathlib import Path
    import pickle
    import os

    # Try to import config
    try:
        import config
        HAS_CONFIG = True
    except ImportError:
        HAS_CONFIG = False
        config = None
    return HAS_CONFIG, Path, config, datetime, go, make_subplots, mo, np, os, pd, pickle, px, pymongo


@app.cell
def _(mo):
    mo.md("""
    # Vector Search Performance Analysis

    Interactive analysis of search results across different query types and nonprofits.
    """)
    return


@app.cell
def _(HAS_CONFIG, config, mo):
    mo.md("""## Configuration""")

    # Use config.py if available, otherwise use manual input
    default_uri = config.MONGODB_URI if HAS_CONFIG and hasattr(config, 'MONGODB_URI') else "mongodb+srv://"
    default_db = config.DATABASE_NAME if HAS_CONFIG and hasattr(config, 'DATABASE_NAME') else "your_database"
    default_collection = config.COLLECTION_NAME if HAS_CONFIG and hasattr(config, 'COLLECTION_NAME') else "search_results"

    # MongoDB connection
    mongo_uri = mo.ui.text(
        label="MongoDB URI",
        value=default_uri,
        kind="password"
    )

    db_name = mo.ui.text(
        label="Database Name",
        value=default_db
    )

    collection_name = mo.ui.text(
        label="Collection Name",
        value=default_collection
    )

    use_cache = mo.ui.checkbox(
        label="Use local cache (faster subsequent loads)",
        value=True
    )
    return collection_name, db_name, mongo_uri, use_cache


@app.cell
def _(collection_name, db_name, mo, mongo_uri, use_cache):
    mo.vstack([
        mongo_uri,
        db_name,
        collection_name,
        use_cache,
        mo.md("*Connection will establish automatically when credentials are entered*")
    ])
    return


@app.cell
def _(collection_name, db_name, mo, mongo_uri, pymongo):
    # Always try to connect when we have valid inputs
    client = None
    collection = None
    db = None
    connection_status = "Not connected"
    doc_count = 0

    try:
        if mongo_uri.value and db_name.value and collection_name.value:
            client = pymongo.MongoClient(
                mongo_uri.value,
                serverSelectionTimeoutMS=5000
            )
            db = client[db_name.value]
            collection = db[collection_name.value]

            # Test connection and get count
            doc_count = collection.count_documents({})
            connection_status = f"✓ Connected! Found {doc_count:,} documents"
    except Exception as e:
        connection_status = f"✗ Connection failed: {str(e)}"
        client = None
        collection = None

    mo.md(f"**Status:** {connection_status}")
    return client, collection, connection_status, db, doc_count


@app.cell
def _(collection, mo, pd, pickle, use_cache):
    from pathlib import Path

    df = None
    load_status = ""
    cache_file = Path("data_cache.pkl")

    # Auto-load data when collection is available
    if collection is not None:
        try:
            # Check if cache exists and should be used
            if use_cache.value and cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    load_status = f"✓ Loaded {len(df):,} records from cache with {len(df.columns)} columns"
                except Exception as cache_error:
                    load_status = f"Cache read failed: {cache_error}. Loading from MongoDB..."
                    df = None

            # If not using cache or cache failed, load from MongoDB
            if df is None:
                # Load all documents
                cursor = collection.find({})
                records = list(cursor)

                # Convert to DataFrame
                df = pd.DataFrame(records)

                # Flatten ObjectId fields
                if '_id' in df.columns:
                    df['_id'] = df['_id'].apply(lambda x: str(x) if hasattr(x, '__str__') else x)
                if 'expected_nonprofit_id' in df.columns:
                    df['expected_nonprofit_id'] = df['expected_nonprofit_id'].apply(
                        lambda x: str(x['$oid']) if isinstance(x, dict) and '$oid' in x else str(x)
                    )
                if 'nonprofit_id' in df.columns:
                    df['nonprofit_id'] = df['nonprofit_id'].apply(
                        lambda x: str(x['$oid']) if isinstance(x, dict) and '$oid' in x else str(x)
                    )
                if 'query_id' in df.columns:
                    df['query_id'] = df['query_id'].apply(
                        lambda x: str(x['$oid']) if isinstance(x, dict) and '$oid' in x else str(x)
                    )

                # Convert date
                if 'evaluated_at' in df.columns:
                    df['evaluated_at'] = pd.to_datetime(
                        df['evaluated_at'].apply(
                            lambda x: x['$date'] if isinstance(x, dict) and '$date' in x else x
                        )
                    )

                # Handle Double type for numeric fields
                numeric_fields = ['reciprocal_rank', 'search_latency_ms']
                for field in numeric_fields:
                    if field in df.columns:
                        df[field] = df[field].apply(
                            lambda x: float(x['$numberDouble']) if isinstance(x, dict) and '$numberDouble' in x else float(x) if x is not None else None
                        )

                load_status = f"✓ Loaded {len(df):,} records from MongoDB with {len(df.columns)} columns"

                # Save to cache if enabled
                if use_cache.value:
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(df, f)
                        load_status += " (cached for next time)"
                    except Exception as cache_error:
                        load_status += f" (cache save failed: {cache_error})"

        except Exception as e:
            load_status = f"✗ Load failed: {str(e)}"
            df = None

    mo.md(f"**Load Status:** {load_status}")
    return cache_file, df, load_status


@app.cell
def _(df, mo):
    if df is not None:
        mo.md(f"""
        ## Dataset Overview

        - **Total records:** {len(df):,}
        - **Date range:** {df['evaluated_at'].min()} to {df['evaluated_at'].max()}
        - **Unique nonprofits:** {df['expected_nonprofit_name'].nunique()}
        - **Query types:** {df['query_type'].nunique()}
        - **Query strategies:** {df['query_strategy'].nunique() if 'query_strategy' in df.columns else 'N/A'}
        """)
    return


@app.cell
def _(df, mo):
    if df is not None:
        # Show sample
        mo.md("### Sample Data")
        display_cols = [
            'query_type', 'expected_nonprofit_name', 'rank',
            'reciprocal_rank', 'found_in_top_5', 'search_latency_ms'
        ]
        available_cols = [col for col in display_cols if col in df.columns]
        mo.ui.table(df[available_cols].head(10))
    return


@app.cell
def _(df, mo):
    mo.md("""## Filters""")

    if df is not None:
        # Create filter widgets
        query_type_filter = mo.ui.multiselect(
            options=sorted(df['query_type'].unique().tolist()),
            label="Query Types",
            value=sorted(df['query_type'].unique().tolist())
        )

        nonprofit_filter = mo.ui.multiselect(
            options=sorted(df['expected_nonprofit_name'].dropna().unique().tolist()),
            label="Nonprofits (leave empty for all)",
            value=[]
        )

        mo.hstack([query_type_filter, nonprofit_filter], justify="start")
    return nonprofit_filter, query_type_filter


@app.cell
def _(df, nonprofit_filter, query_type_filter):
    # Apply filters
    filtered_df = df

    if df is not None:
        # Filter by query type
        if query_type_filter.value:
            filtered_df = filtered_df[filtered_df['query_type'].isin(query_type_filter.value)]

        # Filter by nonprofit
        if nonprofit_filter.value:
            filtered_df = filtered_df[filtered_df['expected_nonprofit_name'].isin(nonprofit_filter.value)]

    filtered_df
    return (filtered_df,)


@app.cell
def _(filtered_df, mo):
    if filtered_df is not None and len(filtered_df) > 0:
        mo.md(f"""
        ### Filtered Dataset
        **{len(filtered_df):,}** records selected
        """)
    return


@app.cell
def _(filtered_df, mo):
    mo.md("""## Aggregate Metrics by Query Type""")

    if filtered_df is not None and len(filtered_df) > 0:
        agg_by_query_type = filtered_df.groupby('query_type').agg({
            'reciprocal_rank': ['mean', 'median', 'std'],
            'rank': ['mean', 'median'],
            'found_in_top_1': 'mean',
            'found_in_top_5': 'mean',
            'found_in_top_10': 'mean',
            'found_in_top_20': 'mean',
            'search_latency_ms': 'mean',
            'query_id': 'count'
        }).round(4)

        agg_by_query_type.columns = [
            'MRR_mean', 'MRR_median', 'MRR_std',
            'Rank_mean', 'Rank_median',
            'Top1_rate', 'Top5_rate', 'Top10_rate', 'Top20_rate',
            'Latency_ms', 'Query_count'
        ]

        mo.ui.table(agg_by_query_type.reset_index(), selection=None)
    return


@app.cell
def _(filtered_df, go, mo, np):
    mo.md("""## Heatmap: Query Type × Nonprofit""")

    if filtered_df is not None and len(filtered_df) > 0:
        # Create pivot table
        pivot_data = filtered_df.pivot_table(
            index='expected_nonprofit_name',
            columns='query_type',
            values='reciprocal_rank',
            aggfunc='mean'
        )

        # Sort by mean reciprocal rank across all query types
        pivot_data['_mean'] = pivot_data.mean(axis=1)
        pivot_data = pivot_data.sort_values('_mean', ascending=False)
        pivot_data = pivot_data.drop('_mean', axis=1)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            text=np.round(pivot_data.values, 3),
            texttemplate='%{text}',
            textfont={"size": 8},
            colorbar=dict(title="Mean Reciprocal Rank"),
            hoverongaps=False
        ))

        fig.update_layout(
            title="Mean Reciprocal Rank by Query Type and Nonprofit",
            xaxis_title="Query Type",
            yaxis_title="Nonprofit",
            height=max(400, len(pivot_data) * 20),
            width=1000
        )

        mo.ui.plotly(fig)
    return


@app.cell
def _(filtered_df, mo, px):
    mo.md("""## Success Rates by Query Type""")

    if filtered_df is not None and len(filtered_df) > 0:
        # Prepare success rate data
        success_cols = ['found_in_top_1', 'found_in_top_5', 'found_in_top_10', 'found_in_top_20', 'found_in_top_50']

        success_data = filtered_df.groupby('query_type')[success_cols].mean().reset_index()
        success_data_melted = success_data.melt(
            id_vars='query_type',
            var_name='Metric',
            value_name='Success_Rate'
        )

        # Clean up metric names
        success_data_melted['Metric'] = success_data_melted['Metric'].str.replace('found_in_', 'Top ')
        success_data_melted['Metric'] = success_data_melted['Metric'].str.replace('_', ' ')

        fig2 = px.bar(
            success_data_melted,
            x='query_type',
            y='Success_Rate',
            color='Metric',
            barmode='group',
            title='Success Rates by Query Type',
            labels={'Success_Rate': 'Success Rate', 'query_type': 'Query Type'}
        )

        fig2.update_layout(height=500, yaxis_tickformat='.0%')

        mo.ui.plotly(fig2)
    return


@app.cell
def _(filtered_df, mo, px):
    mo.md("""## Rank Distribution by Query Type""")

    if filtered_df is not None and len(filtered_df) > 0:
        # Only show ranks where nonprofit was found (not null)
        rank_data = filtered_df[filtered_df['rank'].notna()]

        fig3 = px.box(
            rank_data,
            x='query_type',
            y='rank',
            title='Distribution of Ranks by Query Type',
            labels={'rank': 'Rank', 'query_type': 'Query Type'},
            points='outliers'
        )

        fig3.update_layout(height=500)
        fig3.update_yaxis(range=[0, 50])  # Focus on top 50

        mo.ui.plotly(fig3)
    return


@app.cell
def _(filtered_df, mo, px):
    mo.md("""## Search Latency by Query Type""")

    if filtered_df is not None and len(filtered_df) > 0:
        fig4 = px.box(
            filtered_df,
            x='query_type',
            y='search_latency_ms',
            title='Search Latency Distribution by Query Type',
            labels={'search_latency_ms': 'Latency (ms)', 'query_type': 'Query Type'},
            points='outliers'
        )

        fig4.update_layout(height=500)

        mo.ui.plotly(fig4)
    return


@app.cell
def _(filtered_df, mo):
    mo.md("""## Detailed Query Analysis""")

    if filtered_df is not None and len(filtered_df) > 0:
        # Select specific nonprofit to drill down
        nonprofit_select = mo.ui.dropdown(
            options=sorted(filtered_df['expected_nonprofit_name'].dropna().unique().tolist()),
            label="Select Nonprofit for Details",
            value=sorted(filtered_df['expected_nonprofit_name'].dropna().unique().tolist())[0]
        )

        nonprofit_select
    return (nonprofit_select,)


@app.cell
def _(filtered_df, mo, nonprofit_select):
    if filtered_df is not None and nonprofit_select.value:
        nonprofit_detail = filtered_df[
            filtered_df['expected_nonprofit_name'] == nonprofit_select.value
        ].copy()

        mo.md(f"""
        ### Results for: {nonprofit_select.value}

        **Total queries:** {len(nonprofit_detail)}
        """)

        # Show detailed results
        detail_cols = [
            'query_type', 'query_text', 'rank', 'reciprocal_rank',
            'found_in_top_1', 'found_in_top_5', 'found_in_top_10',
            'search_latency_ms'
        ]
        available_detail_cols = [col for col in detail_cols if col in nonprofit_detail.columns]

        mo.ui.table(
            nonprofit_detail[available_detail_cols].sort_values('reciprocal_rank', ascending=False),
            selection=None
        )
    return


@app.cell
def _(filtered_df, mo):
    if filtered_df is not None and len(filtered_df) > 0:
        mo.md("""
        ---

        ## Export Data

        To export the current filtered data, you can download it as CSV using the dataframe download option, or run this in a Python cell:
        ```python
        filtered_df.to_csv('export.csv', index=False)
        ```
        """)
    return


if __name__ == "__main__":
    app.run()
