import marimo

__generated_with = "0.17.7"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pymongo
    import pandas as pd
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
    return HAS_CONFIG, Path, config, mo, np, pd, pickle, pymongo


@app.cell
def _(mo):
    mo.md("""
    # Nonprofit Embedding Visualizer

    Interactive visualization of nonprofit vector embeddings using dimensionality reduction.
    """)
    return


@app.cell
def _(HAS_CONFIG, config, mo):
    # Use config.py if available
    default_uri = config.MONGODB_URI if HAS_CONFIG and hasattr(config, 'MONGODB_URI') else "mongodb+srv://"
    default_db = config.DATABASE_NAME if HAS_CONFIG and hasattr(config, 'DATABASE_NAME') else "deedserver"
    default_collection = "nonprofits_testset_vectors"

    use_cache = mo.ui.checkbox(
        label="Use local cache",
        value=True
    )

    load_button = mo.ui.run_button(label="Load Nonprofit Embeddings")

    config_display = mo.vstack([
        mo.md("""## Load Data"""),
        use_cache,
        load_button,
        mo.md(f"*Using: {default_db} / {default_collection}*" if HAS_CONFIG else "*No config.py found*")
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
    np,
    pd,
    pickle,
    pymongo,
    use_cache,
):
    # Stop execution until button is clicked
    mo.stop(not load_button.value)

    df_nonprofits = None
    status_msg = ""
    cache_file = Path("embeddings_cache.pkl")

    # Try cache first if enabled
    if use_cache.value and cache_file.exists():
        try:
            status_msg = "Loading from cache..."
            with open(cache_file, 'rb') as f:
                df_nonprofits = pickle.load(f)
            status_msg = f"✓ Loaded {len(df_nonprofits):,} nonprofits from cache"
        except Exception as e:
            status_msg = f"Cache failed ({e}), loading from MongoDB..."
            df_nonprofits = None

    # Load from MongoDB if needed
    if df_nonprofits is None:
        try:
            status_msg = "Connecting to MongoDB..."
            client = pymongo.MongoClient(default_uri, serverSelectionTimeoutMS=10000)
            db = client[default_db]
            collection = db[default_collection]

            status_msg = f"Loading {collection.count_documents({}):,} documents from MongoDB..."
            cursor = collection.find({})
            records = list(cursor)

            status_msg = "Processing embeddings..."

            # Extract embeddings and metadata
            processed_records = []
            for record in records:
                if 'vectors' in record and len(record['vectors']) > 0:
                    # Get the first vector entry (or you can filter by strategy)
                    vector_data = record['vectors'][0]

                    processed_records.append({
                        'nonprofitId': record.get('nonprofitId', 'N/A'),
                        'name': record.get('name', 'N/A'),
                        'categories': ', '.join(record.get('categories', [])),
                        'location_city': record.get('location', {}).get('city', 'N/A'),
                        'location_state': record.get('location', {}).get('stateCode', 'N/A'),
                        'location_country': record.get('location', {}).get('countryCode', 'N/A'),
                        'strategy': vector_data.get('strategy', 'N/A'),
                        'template_text': vector_data.get('template_text', 'N/A'),
                        'embedding': np.array(vector_data.get('embedding', [])),
                        'testset_names': ', '.join(record.get('_testset_names', [])),
                        'testset_strategies': ', '.join(record.get('_testset_strategies', []))
                    })

            df_nonprofits = pd.DataFrame(processed_records)

            # Save to cache
            if use_cache.value and len(df_nonprofits) > 0:
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(df_nonprofits, f)
                    status_msg = f"✓ Loaded {len(df_nonprofits):,} nonprofits from MongoDB (cached)"
                except Exception as cache_error:
                    status_msg = f"✓ Loaded {len(df_nonprofits):,} nonprofits from MongoDB (cache save failed)"
            else:
                status_msg = f"✓ Loaded {len(df_nonprofits):,} nonprofits from MongoDB"

        except Exception as e:
            status_msg = f"✗ Error: {str(e)}"
            df_nonprofits = None

    if status_msg:
        mo.md(f"**Status:** {status_msg}")
    return (df_nonprofits,)


@app.cell
def _(df_nonprofits, mo):
    overview = None
    if df_nonprofits is not None:
        embedding_dim = len(df_nonprofits['embedding'].iloc[0]) if len(df_nonprofits) > 0 else 0
        overview = mo.md(f"""
        ## Dataset Overview

        - **Total nonprofits:** {len(df_nonprofits):,}
        - **Embedding dimensions:** {embedding_dim}
        - **Unique categories:** {df_nonprofits['categories'].nunique()}
        - **Strategies:** {df_nonprofits['strategy'].nunique()}
        - **Testset names:** {df_nonprofits['testset_names'].nunique()}
        - **States:** {df_nonprofits['location_state'].nunique()}
        """)
    overview
    return


@app.cell
def _(df_nonprofits, mo):
    sample_display = None
    if df_nonprofits is not None:
        sample_cols = ['name', 'categories', 'location_city', 'location_state', 'strategy', 'testset_names']
        sample_display = mo.vstack([
            mo.md("### Sample Data"),
            mo.ui.table(df_nonprofits[sample_cols].head(10))
        ])
    sample_display
    return


@app.cell
def _(df_nonprofits, mo):
    # Create filter widgets
    reduction_method = None
    color_by = None
    filter_category = None
    filter_state = None
    filters_display = None

    if df_nonprofits is not None:
        reduction_method = mo.ui.dropdown(
            options=["PCA", "UMAP", "t-SNE"],
            value="PCA",
            label="Dimensionality Reduction Method"
        )

        color_by = mo.ui.dropdown(
            options=["categories", "strategy", "testset_names", "location_state"],
            value="categories",
            label="Color By"
        )

        # Extract all unique individual categories
        all_categories = set()
        for cat_str in df_nonprofits['categories'].dropna():
            cats = [c.strip() for c in cat_str.split(',') if c.strip()]
            all_categories.update(cats)

        filter_category = mo.ui.multiselect(
            options=sorted(list(all_categories)),
            label="Filter by Category (leave empty for all)",
            value=[]
        )

        filter_state = mo.ui.multiselect(
            options=sorted(df_nonprofits['location_state'].dropna().unique().tolist()),
            label="Filter by State (leave empty for all)",
            value=[]
        )

        filters_display = mo.vstack([
            mo.md("""## Visualization Settings"""),
            mo.hstack([reduction_method, color_by], justify="start"),
            mo.hstack([filter_category, filter_state], justify="start")
        ])

    filters_display
    return color_by, filter_category, filter_state, reduction_method


@app.cell
def _(df_nonprofits, filter_category, filter_state):
    # Apply filters
    filtered_nonprofits = df_nonprofits

    if df_nonprofits is not None:
        if filter_category.value:
            # Filter nonprofits that contain ANY of the selected categories
            def has_any_category(cat_str, selected_cats):
                if not cat_str or not isinstance(cat_str, str):
                    return False
                nonprofit_cats = set(c.strip() for c in cat_str.split(',') if c.strip())
                return bool(nonprofit_cats.intersection(selected_cats))

            filtered_nonprofits = filtered_nonprofits[
                filtered_nonprofits['categories'].apply(
                    lambda x: has_any_category(x, set(filter_category.value))
                )
            ]

        if filter_state.value:
            filtered_nonprofits = filtered_nonprofits[
                filtered_nonprofits['location_state'].isin(filter_state.value)
            ]

    return (filtered_nonprofits,)


@app.cell
def _(filtered_nonprofits, mo):
    filtered_status = None
    if filtered_nonprofits is not None and len(filtered_nonprofits) > 0:
        filtered_status = mo.md(f"""
        ### Filtered Dataset
        **{len(filtered_nonprofits):,}** nonprofits selected for visualization
        """)
    filtered_status
    return


@app.cell
def _(filtered_nonprofits, mo, np, pd, reduction_method):
    # Perform dimensionality reduction
    embedding_2d = None
    reduction_status = None

    if filtered_nonprofits is not None and len(filtered_nonprofits) > 0:
        try:
            reduction_status = mo.md(f"Computing {reduction_method.value} reduction...")

            # Stack embeddings into matrix
            embeddings_matrix = np.vstack(filtered_nonprofits['embedding'].values)

            if reduction_method.value == "PCA":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, whiten=True)
                X_embedded = reducer.fit_transform(embeddings_matrix)
                reduction_status = mo.md(f"✓ PCA complete (explained variance: {reducer.explained_variance_ratio_.sum():.2%})")

            elif reduction_method.value == "UMAP":
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42)
                X_embedded = reducer.fit_transform(embeddings_matrix)
                reduction_status = mo.md("✓ UMAP complete")

            elif reduction_method.value == "t-SNE":
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                X_embedded = reducer.fit_transform(embeddings_matrix)
                reduction_status = mo.md("✓ t-SNE complete")

            # Create dataframe with 2D embeddings
            embedding_2d = pd.DataFrame({
                'x': X_embedded[:, 0],
                'y': X_embedded[:, 1],
                'nonprofitId': filtered_nonprofits['nonprofitId'].values,
                'name': filtered_nonprofits['name'].values,
                'categories': filtered_nonprofits['categories'].values,
                'location_city': filtered_nonprofits['location_city'].values,
                'location_state': filtered_nonprofits['location_state'].values,
                'strategy': filtered_nonprofits['strategy'].values,
                'testset_names': filtered_nonprofits['testset_names'].values,
                'template_text': filtered_nonprofits['template_text'].values
            }).reset_index()

        except Exception as e:
            reduction_status = mo.md(f"✗ Error during reduction: {str(e)}")
            embedding_2d = None

    reduction_status
    return (embedding_2d,)


@app.cell
def _(color_by, embedding_2d, mo, np, pd):
    # Create interactive scatter plot
    scatter_chart = None
    chart_display = None

    if embedding_2d is not None and len(embedding_2d) > 0:
        import altair as alt
        import hashlib

        # Function to generate color from category set
        def category_set_to_color(category_str):
            """Convert category string to HSL color based on content similarity"""
            # Sort categories to ensure consistency
            cats = sorted([c.strip() for c in category_str.split(',') if c.strip()])

            if not cats:
                return '#cccccc'  # Gray for empty

            # Hash the sorted category set for consistency
            cat_hash = hashlib.md5(','.join(cats).encode()).hexdigest()

            # Use hash to generate hue (0-360)
            hue = int(cat_hash[:8], 16) % 360

            # Adjust saturation and lightness based on number of categories
            num_cats = len(cats)
            saturation = min(70 + num_cats * 5, 90)  # More categories = more saturated
            lightness = max(40, 60 - num_cats * 3)   # More categories = slightly darker

            return f'hsl({hue}, {saturation}%, {lightness}%)'

        # Add color column if coloring by categories
        plot_data = embedding_2d.copy()

        if color_by.value == 'categories':
            plot_data['color_value'] = plot_data['categories'].apply(category_set_to_color)
            use_custom_colors = True
        else:
            plot_data['color_value'] = plot_data[color_by.value]
            use_custom_colors = False

        # Create selection
        brush = alt.selection_interval()

        # Create base chart with conditional coloring
        if use_custom_colors:
            chart = alt.Chart(plot_data).mark_circle(size=60).encode(
                x=alt.X('x:Q', title='Dimension 1'),
                y=alt.Y('y:Q', title='Dimension 2'),
                color=alt.condition(
                    brush,
                    alt.Color('color_value:N', scale=None, title='Categories', legend=None),
                    alt.value('lightgray')
                ),
                tooltip=[
                    alt.Tooltip('name:N', title='Nonprofit'),
                    alt.Tooltip('categories:N', title='Categories'),
                    alt.Tooltip('location_city:N', title='City'),
                    alt.Tooltip('location_state:N', title='State'),
                    alt.Tooltip('strategy:N', title='Strategy'),
                    alt.Tooltip('testset_names:N', title='Testset')
                ]
            ).add_params(
                brush
            ).properties(
                width=700,
                height=600,
                title=f'Nonprofit Embeddings - Colored by {color_by.value.replace("_", " ").title()}'
            )
        else:
            chart = alt.Chart(plot_data).mark_circle(size=60).encode(
                x=alt.X('x:Q', title='Dimension 1'),
                y=alt.Y('y:Q', title='Dimension 2'),
                color=alt.condition(
                    brush,
                    alt.Color('color_value:N', title=color_by.value.replace('_', ' ').title()),
                    alt.value('lightgray')
                ),
                tooltip=[
                    alt.Tooltip('name:N', title='Nonprofit'),
                    alt.Tooltip('categories:N', title='Categories'),
                    alt.Tooltip('location_city:N', title='City'),
                    alt.Tooltip('location_state:N', title='State'),
                    alt.Tooltip('strategy:N', title='Strategy'),
                    alt.Tooltip('testset_names:N', title='Testset')
                ]
            ).add_params(
                brush
            ).properties(
                width=700,
                height=600,
                title=f'Nonprofit Embeddings - Colored by {color_by.value.replace("_", " ").title()}'
            )

        scatter_chart = mo.ui.altair_chart(chart)

        chart_display = mo.vstack([
            mo.md("## Embedding Visualization"),
            mo.md("*Categories are colored by content similarity - similar category sets have similar colors*") if color_by.value == 'categories' else None,
            scatter_chart
        ])

    chart_display
    return (scatter_chart,)


@app.cell
def _(mo, scatter_chart):
    # Show selected data in table
    selection_table = None

    if scatter_chart is not None and len(scatter_chart.value) > 0:
        display_cols = ['name', 'categories', 'location_city', 'location_state', 'strategy', 'testset_names']
        selection_table = mo.ui.table(
            scatter_chart.value[display_cols],
            selection=None
        )

        mo.vstack([
            mo.md(f"""
            ## Selected Nonprofits
            You've selected **{len(scatter_chart.value)}** nonprofit(s). Click points on the chart to explore!
            """),
            selection_table
        ])
    return (selection_table,)


@app.cell
def _(mo, scatter_chart, selection_table):
    # Show detailed view of selected nonprofit
    detail_view = None

    if scatter_chart is not None and len(scatter_chart.value) > 0:
        # Use table selection if available, otherwise use chart selection
        selected_data = (
            selection_table.value if selection_table is not None and len(selection_table.value) > 0
            else scatter_chart.value
        )

        if len(selected_data) > 0:
            # Show first selected nonprofit's details
            first_selected = selected_data.iloc[0]

            detail_view = mo.md(f"""
            ---
            ## Nonprofit Details

            **Name:** {first_selected['name']}

            **Categories:** {first_selected['categories']}

            **Location:** {first_selected['location_city']}, {first_selected['location_state']}

            **Strategy:** {first_selected['strategy']}

            **Testset:** {first_selected['testset_names']}

            **Template Text:**
            ```
            {first_selected['template_text']}
            ```
            """)

    detail_view
    return (detail_view,)


@app.cell
def _(embedding_2d, mo):
    export_info = None
    if embedding_2d is not None and len(embedding_2d) > 0:
        export_info = mo.md("""
        ---

        ## Export Data

        To export the 2D embeddings, run:
        ```python
        embedding_2d.to_csv('embeddings_2d.csv', index=False)
        ```
        """)
    export_info
    return


if __name__ == "__main__":
    app.run()
