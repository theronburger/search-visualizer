import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pymongo
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    from pathlib import Path
    import pickle

    # Try to import config
    try:
        import config
        HAS_CONFIG = True
    except ImportError:
        HAS_CONFIG = False
        config = None
    return HAS_CONFIG, Path, config, go, mo, np, pd, pickle, px, pymongo


@app.cell
def _(mo):
    mo.md("""
    # Vector Search Performance Analysis

    Interactive analysis of search results across different query types and nonprofits.
    """)
    return


@app.cell
def _(HAS_CONFIG, config, mo):
    # Use config.py if available
    default_uri = config.MONGODB_URI if HAS_CONFIG and hasattr(config, 'MONGODB_URI') else "mongodb+srv://"
    default_db = config.DATABASE_NAME if HAS_CONFIG and hasattr(config, 'DATABASE_NAME') else "your_database"
    default_collection = config.COLLECTION_NAME if HAS_CONFIG and hasattr(config, 'COLLECTION_NAME') else "search_results"

    use_cache = mo.ui.checkbox(
        label="Use local cache",
        value=True
    )

    load_button = mo.ui.run_button(label="Load Data")

    config_display = mo.vstack([
        mo.md("""## Load Data"""),
        use_cache,
        load_button,
        mo.md(f"*Using config: {default_db} / {default_collection}*" if HAS_CONFIG else "*No config.py found*")
    ])

    config_display
    return default_collection, default_db, default_uri, load_button, use_cache


@app.cell
def _(
    Path,
    default_collection,
    default_db,
    default_uri,
    load_button,
    mo,
    pd,
    pickle,
    pymongo,
    use_cache,
):
    # Stop execution until button is clicked
    mo.stop(not load_button.value)

    df_final = None
    status_msg = ""
    cache_file = Path("data_cache.pkl")

    # Button was clicked, load data
    # Try cache first if enabled
    if use_cache.value and cache_file.exists():
        try:
            status_msg = "Loading from cache..."
            with open(cache_file, 'rb') as f:
                df_final = pickle.load(f)
            status_msg = f"✓ Loaded {len(df_final):,} records from cache"
        except Exception as e:
            status_msg = f"Cache failed ({e}), loading from MongoDB..."
            df_final = None

    # Load from MongoDB if needed
    if df_final is None:
        try:
            status_msg = "Connecting to MongoDB..."
            client = pymongo.MongoClient(default_uri, serverSelectionTimeoutMS=10000)
            db = client[default_db]
            collection = db[default_collection]

            status_msg = f"Loading {collection.count_documents({}):,} documents from MongoDB..."
            cursor = collection.find({})
            records = list(cursor)

            status_msg = "Processing data..."
            df_final = pd.DataFrame(records)

            # Flatten ObjectId fields
            if '_id' in df_final.columns:
                df_final['_id'] = df_final['_id'].apply(lambda x: str(x) if hasattr(x, '__str__') else x)
            if 'expected_nonprofit_id' in df_final.columns:
                df_final['expected_nonprofit_id'] = df_final['expected_nonprofit_id'].apply(
                    lambda x: str(x['$oid']) if isinstance(x, dict) and '$oid' in x else str(x)
                )
            if 'nonprofit_id' in df_final.columns:
                df_final['nonprofit_id'] = df_final['nonprofit_id'].apply(
                    lambda x: str(x['$oid']) if isinstance(x, dict) and '$oid' in x else str(x)
                )
            if 'query_id' in df_final.columns:
                df_final['query_id'] = df_final['query_id'].apply(
                    lambda x: str(x['$oid']) if isinstance(x, dict) and '$oid' in x else str(x)
                )

            # Convert date
            if 'evaluated_at' in df_final.columns:
                df_final['evaluated_at'] = pd.to_datetime(
                    df_final['evaluated_at'].apply(
                        lambda x: x['$date'] if isinstance(x, dict) and '$date' in x else x
                    )
                )

            # Handle Double type for numeric fields
            numeric_fields = ['reciprocal_rank', 'search_latency_ms']
            for field in numeric_fields:
                if field in df_final.columns:
                    df_final[field] = df_final[field].apply(
                        lambda x: float(x['$numberDouble']) if isinstance(x, dict) and '$numberDouble' in x else float(x) if x is not None else None
                    )

            # Save to cache
            if use_cache.value:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df_final, f)
                    status_msg = f"✓ Loaded {len(df_final):,} records from MongoDB (cached)"
                except Exception as cache_error:
                    status_msg = f"✓ Loaded {len(df_final):,} records from MongoDB (cache save failed)"
            else:
                status_msg = f"✓ Loaded {len(df_final):,} records from MongoDB"

        except Exception as e:
            status_msg = f"✗ Error: {str(e)}"
            df_final = None

    if status_msg:
        mo.md(f"**Status:** {status_msg}")
    return (df_final,)


@app.cell
def _(df_final, mo):
    overview = None
    if df_final is not None:
        overview = mo.md(f"""
        ## Dataset Overview

        - **Total records:** {len(df_final):,}
        - **Date range:** {df_final['evaluated_at'].min()} to {df_final['evaluated_at'].max()}
        - **Unique nonprofits:** {df_final['expected_nonprofit_name'].nunique()}
        - **Query types:** {df_final['query_type'].nunique()}
        - **Query strategies:** {df_final['query_strategy'].nunique() if 'query_strategy' in df_final.columns else 'N/A'}
        """)
    overview
    return


@app.cell
def _(df_final, mo):
    sample_display = None
    if df_final is not None:
        display_cols = [
            'query_type', 'expected_nonprofit_name', 'rank',
            'reciprocal_rank', 'found_in_top_5', 'search_latency_ms'
        ]
        available_cols = [col for col in display_cols if col in df_final.columns]
        sample_display = mo.vstack([
            mo.md("### Sample Data"),
            mo.ui.table(df_final[available_cols].head(10))
        ])
    sample_display
    return


@app.cell
def _(df_final, mo):
    query_type_filter = None
    nonprofit_filter = None
    filters_display = None

    if df_final is not None:
        # Create filter widgets
        query_type_filter = mo.ui.multiselect(
            options=sorted(df_final['query_type'].unique().tolist()),
            label="Query Types",
            value=sorted(df_final['query_type'].unique().tolist())
        )

        nonprofit_filter = mo.ui.multiselect(
            options=sorted(df_final['expected_nonprofit_name'].dropna().unique().tolist()),
            label="Nonprofits (leave empty for all)",
            value=[]
        )

        filters_display = mo.vstack([
            mo.md("""## Filters"""),
            mo.hstack([query_type_filter, nonprofit_filter], justify="start")
        ])

    filters_display
    return nonprofit_filter, query_type_filter


@app.cell
def _(df_final, nonprofit_filter, query_type_filter):
    # Apply filters
    filtered_df = df_final

    if df_final is not None:
        # Filter by query type
        if query_type_filter.value:
            filtered_df = filtered_df[filtered_df['query_type'].isin(query_type_filter.value)]

        # Filter by nonprofit
        if nonprofit_filter.value:
            filtered_df = filtered_df[filtered_df['expected_nonprofit_name'].isin(nonprofit_filter.value)]
    return (filtered_df,)


@app.cell
def _(filtered_df, mo):
    filtered_status = None
    if filtered_df is not None and len(filtered_df) > 0:
        filtered_status = mo.md(f"""
        ### Filtered Dataset
        **{len(filtered_df):,}** records selected
        """)
    filtered_status
    return


@app.cell
def _(filtered_df, mo):
    agg_display = None
    agg_by_query_type = None

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

        agg_display = mo.vstack([
            mo.md("""## Aggregate Metrics by Query Type"""),
            mo.ui.table(agg_by_query_type.reset_index(), selection=None)
        ])

    agg_display
    return


@app.cell
def _(filtered_df, go, mo, np, pd):
    heatmap_display = None
    heatmap_chart = None
    pivot_data = None

    if filtered_df is not None and len(filtered_df) > 0:
        # Create pivot table
        pivot_data = filtered_df.pivot_table(
            index='expected_nonprofit_name',
            columns='query_type',
            values='reciprocal_rank',
            aggfunc='mean'
        )

        # Sort rows (nonprofits) by mean reciprocal rank across all query types
        pivot_data['_mean_row'] = pivot_data.mean(axis=1)
        pivot_data = pivot_data.sort_values('_mean_row', ascending=False)
        pivot_data = pivot_data.drop('_mean_row', axis=1)

        # Sort columns (query types) by mean reciprocal rank across all nonprofits
        col_means = pivot_data.mean(axis=0).sort_values(ascending=False)
        pivot_data = pivot_data[col_means.index]

        # Create text labels - show value or empty string for NaN
        text_labels = np.where(
            np.isnan(pivot_data.values),
            '',
            np.round(pivot_data.values, 3).astype(str)
        )

        # Calculate detailed stats for hover
        hover_text = []
        for i, nonprofit in enumerate(pivot_data.index):
            row = []
            for j, query_type in enumerate(pivot_data.columns):
                mrr = pivot_data.iloc[i, j]
                if pd.notna(mrr):
                    # Get stats for this cell
                    _cell_data = filtered_df[
                        (filtered_df['query_type'] == query_type) &
                        (filtered_df['expected_nonprofit_name'] == nonprofit)
                    ]
                    if len(_cell_data) > 0:
                        # Get first query result for this combo
                        _first = _cell_data.iloc[0]
                        _nonprofit_id = _first['expected_nonprofit_id']
                        _query_text = _first.get('query_text', 'N/A')
                        _avg_rank = _cell_data['rank'].mean()
                        _mrr5 = _cell_data['found_in_top_5'].mean()
                        _mrr10 = _cell_data['found_in_top_10'].mean()
                        _mrr20 = _cell_data['found_in_top_20'].mean()
                        _avg_latency = _cell_data['search_latency_ms'].mean()

                        # Get top results from the first query
                        _top_results = _first.get('top_k_results', [])[:10] if 'top_k_results' in _first else []

                        # Build hover text
                        hover = f"<b>{nonprofit}</b> (ID: {_nonprofit_id})<br>"
                        hover += f"<b>Query Type:</b> {query_type}<br>"
                        hover += f"<i>\"{_query_text}\"</i><br><br>"
                        hover += f"<b>Rank:</b> {_avg_rank:.1f}<br>"
                        hover += f"<b>MRR@5:</b> {_mrr5:.2f}  <b>MRR@10:</b> {_mrr10:.2f}  <b>MRR@20:</b> {_mrr20:.2f}<br><br>"

                        if _top_results:
                            hover += "<b>Top Results:</b><br>"
                            for idx, result in enumerate(_top_results, 1):
                                _res_id = result.get('nonprofit_id', {}).get('$oid', 'N/A') if isinstance(result.get('nonprofit_id'), dict) else result.get('nonprofit_id', 'N/A')
                                _res_name = result.get('nonprofit_name', 'N/A')
                                hover += f"{idx}. {_res_name[:30]}... ({_res_id})<br>"
                            hover += "<br>"

                        hover += f"<b>Latency:</b> {_avg_latency:.0f}ms"
                    else:
                        hover = f"<b>{nonprofit}</b><br>{query_type}<br><br>MRR: {mrr:.3f}"
                else:
                    hover = f"<b>{nonprofit}</b><br>{query_type}<br><br>No data"
                row.append(hover)
            hover_text.append(row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='RdYlGn',
            text=text_labels,
            hovertext=hover_text,
            hovertemplate='%{hovertext}<extra></extra>',
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Mean<br>Reciprocal<br>Rank"),
            hoverongaps=False,
            xgap=3,
            ygap=1
        ))

        fig.update_layout(
            title="Mean Reciprocal Rank by Query Type and Nonprofit<br><sub>Sorted by performance (best at top-left). Hover for details.</sub>",
            xaxis=dict(
                title="Query Type",
                side='top',  # Move x-axis labels to top
                tickangle=-45,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                title="Nonprofit",
                tickfont=dict(size=10)
            ),
            height=max(500, len(pivot_data) * 22),
            width=max(800, len(pivot_data.columns) * 100),
            margin=dict(t=200, b=50, l=200, r=100)  # More space for labels
        )

        heatmap_display = mo.vstack([
            mo.md("""## Heatmap: Query Type × Nonprofit"""),
            mo.ui.plotly(fig)
        ])

    heatmap_display
    return


@app.cell
def _(filtered_df, mo, px):
    success_display = None
    fig2 = None
    success_cols = None
    success_data = None
    success_data_melted = None

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

        success_display = mo.vstack([
            mo.md("""## Success Rates by Query Type"""),
            mo.ui.plotly(fig2)
        ])

    success_display
    return


@app.cell
def _(filtered_df, mo, px):
    rank_display = None
    fig3 = None
    rank_data = None

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
        fig3.update_yaxes(range=[0, 50])  # Focus on top 50

        rank_display = mo.vstack([
            mo.md("""## Rank Distribution by Query Type"""),
            mo.ui.plotly(fig3)
        ])

    rank_display
    return


@app.cell
def _(filtered_df, mo, px):
    latency_display = None
    fig4 = None

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

        latency_display = mo.vstack([
            mo.md("""## Search Latency by Query Type"""),
            mo.ui.plotly(fig4)
        ])

    latency_display
    return


@app.cell
def _(filtered_df, mo):
    export_info = None
    if filtered_df is not None and len(filtered_df) > 0:
        export_info = mo.md("""
        ---

        ## Export Data

        To export the current filtered data, run:
        ```python
        filtered_df.to_csv('export.csv', index=False)
        ```
        """)
    export_info
    return


if __name__ == "__main__":
    app.run()
