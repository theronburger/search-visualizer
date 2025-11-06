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
    clustering_method = None
    n_clusters = None
    eps = None
    min_samples = None
    filters_display = None

    if df_nonprofits is not None:
        reduction_method = mo.ui.dropdown(
            options=["PCA", "UMAP", "t-SNE"],
            value="PCA",
            label="Dimensionality Reduction Method"
        )

        clustering_method = mo.ui.dropdown(
            options=["None", "K-Means", "DBSCAN", "HDBSCAN"],
            value="None",
            label="Clustering Method"
        )

        # K-Means parameters
        n_clusters = mo.ui.slider(
            start=2,
            stop=100,
            value=5,
            label="K-Means: Number of Clusters"
        )

        kmeans_init = mo.ui.dropdown(
            options=["k-means++", "random"],
            value="k-means++",
            label="K-Means: Initialization Method"
        )

        kmeans_n_init = mo.ui.slider(
            start=1,
            stop=50,
            value=10,
            label="K-Means: Number of Initializations"
        )

        kmeans_max_iter = mo.ui.slider(
            start=100,
            stop=1000,
            step=100,
            value=300,
            label="K-Means: Max Iterations"
        )

        # DBSCAN parameters
        eps = mo.ui.slider(
            start=0.1,
            stop=2.0,
            step=0.1,
            value=0.5,
            label="DBSCAN: eps (neighborhood size)"
        )

        min_samples = mo.ui.slider(
            start=2,
            stop=50,
            value=5,
            label="DBSCAN/HDBSCAN: min_samples"
        )

        color_by = mo.ui.dropdown(
            options=["categories", "strategy", "testset_names", "location_state", "cluster"],
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
            mo.md("### Clustering"),
            clustering_method,
            mo.md("#### K-Means Parameters"),
            mo.hstack([n_clusters, kmeans_init], justify="start"),
            mo.hstack([kmeans_n_init, kmeans_max_iter], justify="start"),
            mo.md("#### DBSCAN/HDBSCAN Parameters"),
            mo.hstack([eps, min_samples], justify="start"),
            mo.md("### Filters"),
            mo.hstack([filter_category, filter_state], justify="start")
        ])

    filters_display
    return clustering_method, color_by, eps, filter_category, filter_state, kmeans_init, kmeans_max_iter, kmeans_n_init, min_samples, n_clusters, reduction_method


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
def _(clustering_method, eps, filtered_nonprofits, kmeans_init, kmeans_max_iter, kmeans_n_init, min_samples, mo, n_clusters, np):
    # Perform clustering on high-dimensional embeddings
    clustered_nonprofits = None
    cluster_status = None

    if filtered_nonprofits is not None and len(filtered_nonprofits) > 0:
        clustered_nonprofits = filtered_nonprofits.copy()

        if clustering_method.value != "None":
            try:
                cluster_status = mo.md(f"Computing {clustering_method.value} clustering...")

                # Stack embeddings into matrix
                embeddings_cluster = np.vstack(filtered_nonprofits['embedding'].values)

                if clustering_method.value == "K-Means":
                    from sklearn.cluster import KMeans
                    clusterer = KMeans(
                        n_clusters=n_clusters.value,
                        init=kmeans_init.value,
                        n_init=kmeans_n_init.value,
                        max_iter=kmeans_max_iter.value,
                        random_state=42
                    )
                    cluster_labels = clusterer.fit_predict(embeddings_cluster)
                    n_clusters_found = n_clusters.value
                    inertia = clusterer.inertia_
                    cluster_status = mo.md(f"✓ K-Means complete ({n_clusters_found} clusters, inertia: {inertia:.2f})")

                elif clustering_method.value == "DBSCAN":
                    from sklearn.cluster import DBSCAN
                    clusterer = DBSCAN(eps=eps.value, min_samples=min_samples.value, metric='cosine')
                    cluster_labels = clusterer.fit_predict(embeddings_cluster)
                    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    cluster_status = mo.md(f"✓ DBSCAN complete ({n_clusters_found} clusters, {n_noise} noise points)")

                elif clustering_method.value == "HDBSCAN":
                    import hdbscan
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_samples.value, metric='euclidean')
                    cluster_labels = clusterer.fit_predict(embeddings_cluster)
                    n_clusters_found = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    n_noise = list(cluster_labels).count(-1)
                    cluster_status = mo.md(f"✓ HDBSCAN complete ({n_clusters_found} clusters, {n_noise} noise points)")

                # Add cluster labels to dataframe (-1 for noise becomes "Noise")
                clustered_nonprofits['cluster'] = [f"Cluster {c}" if c >= 0 else "Noise" for c in cluster_labels]

            except Exception as e:
                cluster_status = mo.md(f"✗ Clustering error: {str(e)}")
                clustered_nonprofits['cluster'] = "No Cluster"
        else:
            clustered_nonprofits['cluster'] = "No Clustering"
            cluster_status = mo.md(f"**{len(clustered_nonprofits):,}** nonprofits selected for visualization")

    cluster_status
    return (clustered_nonprofits,)


@app.cell
def _(clustered_nonprofits, mo, np, pd, reduction_method):
    # Perform dimensionality reduction
    embedding_2d = None
    reduction_status = None

    if clustered_nonprofits is not None and len(clustered_nonprofits) > 0:
        try:
            reduction_status = mo.md(f"Computing {reduction_method.value} reduction...")

            # Stack embeddings into matrix
            embeddings_reduce = np.vstack(clustered_nonprofits['embedding'].values)

            if reduction_method.value == "PCA":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, whiten=True)
                X_embedded = reducer.fit_transform(embeddings_reduce)
                reduction_status = mo.md(f"✓ PCA complete (explained variance: {reducer.explained_variance_ratio_.sum():.2%})")

            elif reduction_method.value == "UMAP":
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42)
                X_embedded = reducer.fit_transform(embeddings_reduce)
                reduction_status = mo.md("✓ UMAP complete")

            elif reduction_method.value == "t-SNE":
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42)
                X_embedded = reducer.fit_transform(embeddings_reduce)
                reduction_status = mo.md("✓ t-SNE complete")

            # Create dataframe with 2D embeddings
            embedding_2d = pd.DataFrame({
                'x': X_embedded[:, 0],
                'y': X_embedded[:, 1],
                'nonprofitId': clustered_nonprofits['nonprofitId'].values,
                'name': clustered_nonprofits['name'].values,
                'categories': clustered_nonprofits['categories'].values,
                'location_city': clustered_nonprofits['location_city'].values,
                'location_state': clustered_nonprofits['location_state'].values,
                'strategy': clustered_nonprofits['strategy'].values,
                'testset_names': clustered_nonprofits['testset_names'].values,
                'template_text': clustered_nonprofits['template_text'].values,
                'cluster': clustered_nonprofits['cluster'].values
            }).reset_index()

        except Exception as e:
            reduction_status = mo.md(f"✗ Error during reduction: {str(e)}")
            embedding_2d = None

    reduction_status
    return (embedding_2d,)


@app.cell
def _(cluster_stats_table, color_by, embedding_2d, mo, np, pd):
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

        # Function to generate color from cluster ID
        def cluster_to_color(cluster_str):
            """Convert cluster string to distinct color using spectrum"""
            if cluster_str == "Noise" or cluster_str == "No Clustering" or cluster_str == "No Cluster":
                return '#888888'  # Gray for noise/no cluster

            # Extract cluster number
            try:
                cluster_num = int(cluster_str.split()[-1])
            except:
                cluster_num = hash(cluster_str) % 20

            # Use golden angle to spread colors evenly across hue spectrum
            golden_angle = 137.508
            hue = (cluster_num * golden_angle) % 360

            # Use high saturation and medium lightness for vibrant, distinct colors
            saturation = 75
            lightness = 55

            return f'hsl({hue}, {saturation}%, {lightness}%)'

        # Add color column based on what we're coloring by
        plot_data = embedding_2d.copy()

        # Check if a cluster is selected from the stats table
        highlighted_cluster = None
        if cluster_stats_table is not None and len(cluster_stats_table.value) > 0:
            highlighted_cluster = cluster_stats_table.value.iloc[0]['Cluster']

        if color_by.value == 'categories':
            plot_data['color_value'] = plot_data['categories'].apply(category_set_to_color)
            use_custom_colors = True
        elif color_by.value == 'cluster':
            plot_data['color_value'] = plot_data['cluster'].apply(cluster_to_color)
            use_custom_colors = True
        else:
            plot_data['color_value'] = plot_data[color_by.value]
            use_custom_colors = False

        # Add opacity column for highlighting selected cluster
        if highlighted_cluster is not None:
            plot_data['opacity'] = plot_data['cluster'].apply(lambda x: 1.0 if x == highlighted_cluster else 0.2)
        else:
            plot_data['opacity'] = 1.0

        # Create selection
        brush = alt.selection_interval()

        # Create base chart with conditional coloring
        if use_custom_colors:
            color_title = color_by.value.replace('_', ' ').title()
            chart = alt.Chart(plot_data).mark_circle(size=60).encode(
                x=alt.X('x:Q', title='Dimension 1'),
                y=alt.Y('y:Q', title='Dimension 2'),
                color=alt.condition(
                    brush,
                    alt.Color('color_value:N', scale=None, title=color_title, legend=None),
                    alt.value('lightgray')
                ),
                opacity=alt.Opacity('opacity:Q', scale=None, legend=None),
                tooltip=[
                    alt.Tooltip('name:N', title='Nonprofit'),
                    alt.Tooltip('categories:N', title='Categories'),
                    alt.Tooltip('location_city:N', title='City'),
                    alt.Tooltip('location_state:N', title='State'),
                    alt.Tooltip('strategy:N', title='Strategy'),
                    alt.Tooltip('testset_names:N', title='Testset'),
                    alt.Tooltip('cluster:N', title='Cluster')
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
                opacity=alt.Opacity('opacity:Q', scale=None, legend=None),
                tooltip=[
                    alt.Tooltip('name:N', title='Nonprofit'),
                    alt.Tooltip('categories:N', title='Categories'),
                    alt.Tooltip('location_city:N', title='City'),
                    alt.Tooltip('location_state:N', title='State'),
                    alt.Tooltip('strategy:N', title='Strategy'),
                    alt.Tooltip('testset_names:N', title='Testset'),
                    alt.Tooltip('cluster:N', title='Cluster')
                ]
            ).add_params(
                brush
            ).properties(
                width=700,
                height=600,
                title=f'Nonprofit Embeddings - Colored by {color_by.value.replace("_", " ").title()}'
            )

        scatter_chart = mo.ui.altair_chart(chart)

        subtitle = None
        if color_by.value == 'categories':
            subtitle = mo.md("*Categories are colored by content similarity - similar category sets have similar colors*")
        elif color_by.value == 'cluster' and highlighted_cluster:
            subtitle = mo.md(f"*Highlighting **{highlighted_cluster}** (selected from Cluster Statistics below). Click a different cluster row to change.*")
        elif color_by.value == 'cluster':
            subtitle = mo.md("*Clusters use golden angle color spacing for maximum distinctiveness. Select a cluster in the table below to highlight it.*")

        chart_display = mo.vstack([
            mo.md("## Embedding Visualization"),
            subtitle,
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

            **Cluster:** {first_selected['cluster']}

            **Template Text:**
            ```
            {first_selected['template_text']}
            ```
            """)

    detail_view
    return (detail_view,)


@app.cell
def _(clustering_method, embedding_2d, mo, pd):
    # Show cluster statistics
    cluster_stats_table = None
    cluster_stats_display = None

    if embedding_2d is not None and len(embedding_2d) > 0 and clustering_method.value != "None":
        cluster_counts = embedding_2d['cluster'].value_counts().sort_index()

        # Calculate stats per cluster
        cluster_data = []
        for cluster_name in sorted(embedding_2d['cluster'].unique()):
            cluster_df = embedding_2d[embedding_2d['cluster'] == cluster_name]
            top_categories = cluster_df['categories'].value_counts().head(3)

            cluster_data.append({
                'Cluster': cluster_name,
                'Size': len(cluster_df),
                'Top Categories': ', '.join([f"{cat} ({count})" for cat, count in top_categories.items()][:3])
            })

        cluster_stats_df = pd.DataFrame(cluster_data)

        cluster_stats_table = mo.ui.table(
            cluster_stats_df,
            selection='single',
            label="Click a cluster to highlight it on the chart"
        )

        cluster_stats_display = mo.vstack([
            mo.md("""
            ---
            ## Cluster Statistics
            *Click a cluster row to highlight it on the chart and see detailed insights*
            """),
            cluster_stats_table
        ])

    cluster_stats_display
    return (cluster_stats_table,)


@app.cell
def _(cluster_stats_table, clustering_method, embedding_2d, mo):
    # Show detailed cluster insights when a cluster is selected
    cluster_insights = None

    if cluster_stats_table is not None and len(cluster_stats_table.value) > 0 and embedding_2d is not None:
        selected_cluster_name = cluster_stats_table.value.iloc[0]['Cluster']
        cluster_members = embedding_2d[embedding_2d['cluster'] == selected_cluster_name]

        # Calculate detailed statistics
        top_5_categories = cluster_members['categories'].value_counts().head(5)
        top_5_states = cluster_members['location_state'].value_counts().head(5)

        # Sample nonprofits
        sample_nonprofits = cluster_members[['name', 'categories', 'location_city', 'location_state']].head(10)

        insights_text = f"""
        ---
        ## 🔍 Cluster Insights: {selected_cluster_name}

        **Size:** {len(cluster_members):,} nonprofits ({len(cluster_members)/len(embedding_2d)*100:.1f}% of dataset)

        ### Top Categories
        {chr(10).join([f"- **{cat}**: {count} nonprofits ({count/len(cluster_members)*100:.1f}%)" for cat, count in top_5_categories.items()])}

        ### Geographic Distribution
        {chr(10).join([f"- **{state}**: {count} nonprofits" for state, count in top_5_states.items()])}

        ### Sample Nonprofits in this Cluster
        """

        cluster_insights = mo.vstack([
            mo.md(insights_text),
            mo.ui.table(sample_nonprofits, selection=None)
        ])

    cluster_insights
    return


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
