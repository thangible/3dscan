import numpy as np
import base64
import io
import os
import json
import glob
from PIL import Image
from dash import Dash, dcc, html, Input, Output, State, no_update, callback
import plotly.graph_objects as go
import plotly.express as px
import dash

class ClusterVisualizationDashboard:
    def __init__(self, clustering_results_dir="exp/clustering_results"):
        self.clustering_results_dir = clustering_results_dir
        self.visualization_dir = os.path.join(clustering_results_dir, "visualization")
        self.current_viz_data = None
        
        # Load available clustering results
        self.available_results = self.load_available_results()
        
        if not self.available_results:
            print("❌ No visualization files found!")
            print("Please run: python prepare_visualization_data.py --all")
    
    def load_available_results(self):
        """Load information about all available visualization files"""
        # Check if summary exists for metadata
        summary_path = os.path.join(self.clustering_results_dir, "clustering_summary.json")
        summary_data = {}
        
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
        
        results = {}
        
        # Find all visualization files
        if not os.path.exists(self.visualization_dir):
            print(f"Visualization directory not found: {self.visualization_dir}")
            return results
        
        viz_files = glob.glob(os.path.join(self.visualization_dir, "visualization_data_*.npy"))
        
        for viz_file in viz_files:
            filename = os.path.basename(viz_file)
            # Extract experiment name from filename
            experiment_name = filename.replace("visualization_data_", "").replace(".npy", "")
            
            # Parse algorithm and parameters from experiment name
            if experiment_name.startswith("kmeans_k"):
                algorithm = "K-means"
                k = int(experiment_name.replace("kmeans_k", ""))
                params = {'k': k}
                display_name = f"K-means (k={k})"
                
                # Try to get metrics from summary
                metrics = self.get_metrics_from_summary(summary_data, 'kmeans', params)
                
            elif experiment_name.startswith("dbscan_eps"):
                algorithm = "DBSCAN"
                # Parse eps and min_samples from filename like "dbscan_eps0.5_min5"
                parts = experiment_name.replace("dbscan_eps", "").split("_min")
                eps = float(parts[0])
                min_samples = int(parts[1])
                params = {'eps': eps, 'min_samples': min_samples}
                display_name = f"DBSCAN (eps={eps}, min_samples={min_samples})"
                
                # Try to get metrics from summary
                metrics = self.get_metrics_from_summary(summary_data, 'dbscan', params)
            else:
                continue  # Skip unknown formats
            
            results[experiment_name] = {
                'algorithm': algorithm,
                'params': params,
                'display_name': display_name,
                'file_path': viz_file,
                'metrics': metrics
            }
        
        print(f"Found {len(results)} visualization files")
        return results
    
    def get_metrics_from_summary(self, summary_data, algorithm, params):
        """Extract metrics from clustering summary if available"""
        metrics = {'silhouette': None, 'n_clusters': None, 'n_noise': 0}
        
        if not summary_data:
            return metrics
        
        results_key = f'{algorithm}_results'
        if results_key in summary_data:
            for result in summary_data[results_key]:
                if result.get('params') == params:
                    metrics.update({
                        'silhouette': result.get('silhouette_score'),
                        'n_clusters': result.get('n_clusters'),
                        'n_noise': result.get('n_noise', 0),
                        'calinski_harabasz': result.get('calinski_harabasz_score'),
                        'davies_bouldin': result.get('davies_bouldin_score')
                    })
                    break
        
        return metrics
    
    def load_visualization_data(self, experiment_key):
        """Load visualization data for a specific clustering experiment"""
        if experiment_key not in self.available_results:
            print(f"Experiment key not found: {experiment_key}")
            return None
        
        file_path = self.available_results[experiment_key]['file_path']
        
        try:
            viz_data = np.load(file_path, allow_pickle=True).item()
            return viz_data
        except Exception as e:
            print(f"Error loading visualization data from {file_path}: {e}")
            return None
    
    def encode_image(self, image_path):
        """Convert image to base64 for display in tooltip"""
        try:
            with Image.open(image_path) as img:
                img.thumbnail((200, 200))
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                encoded_image = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/png;base64,{encoded_image}"
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def create_scatter_plot(self, embeddings, clusters, is_noise, title, is_3d=False):
        """Create 2D or 3D scatter plot with proper noise handling"""
        if embeddings is None or len(embeddings) == 0:
            return go.Figure().add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Handle NaN values in embeddings (from t-SNE with noise points)
        valid_mask = ~np.isnan(embeddings).any(axis=1)
        valid_embeddings = embeddings[valid_mask]
        valid_clusters = clusters[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_embeddings) == 0:
            return go.Figure().add_annotation(
                text="No valid data points",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Handle noise points for DBSCAN
        unique_clusters = np.unique(valid_clusters)
        if -1 in unique_clusters:
            # DBSCAN with noise points
            valid_clusters_no_noise = unique_clusters[unique_clusters != -1]
            colors = px.colors.qualitative.Set3[:len(valid_clusters_no_noise)]
            color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(valid_clusters_no_noise)}
            color_map[-1] = 'lightgray'  # Noise points in light gray
        else:
            # K-means or DBSCAN without noise
            colors = px.colors.qualitative.Set3[:len(unique_clusters)]
            color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
        
        point_colors = [color_map[cluster] for cluster in valid_clusters]
        
        if is_3d:
            fig = go.Figure(data=[go.Scatter3d(
                x=valid_embeddings[:, 0],
                y=valid_embeddings[:, 1],
                z=valid_embeddings[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=point_colors,
                    opacity=0.7,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                text=[f"Cluster {c}" if c != -1 else "Noise" for c in valid_clusters],
                customdata=valid_indices,  # Store original indices
                hovertemplate="<b>%{text}</b><br>" +
                            "X: %{x:.2f}<br>" +
                            "Y: %{y:.2f}<br>" +
                            "Z: %{z:.2f}<extra></extra>"
            )])
            
            # Use autosize and margins instead of hard-coded height so the graph fills its container
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    zaxis_title="Component 3"
                ),
                autosize=True,
                margin=dict(l=20, r=20, t=60, b=20),
                template='plotly_white',
                uirevision='constant'
            )
        else:
            fig = go.Figure(data=[go.Scatter(
                x=valid_embeddings[:, 0],
                y=valid_embeddings[:, 1],
                mode='markers',
                marker=dict(
                    size=6,
                    color=point_colors,
                    opacity=0.7,
                    line=dict(width=0.5, color='DarkSlateGrey')
                ),
                text=[f"Cluster {c}" if c != -1 else "Noise" for c in valid_clusters],
                customdata=valid_indices,  # Store original indices
                hovertemplate="<b>%{text}</b><br>" +
                            "X: %{x:.2f}<br>" +
                            "Y: %{y:.2f}<extra></extra>"
            )])
            
            # Use autosize and margins instead of hard-coded height so the graph fills its container
            fig.update_layout(
                title=title,
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                autosize=True,
                margin=dict(l=20, r=20, t=60, b=20),
                template='plotly_white',
                uirevision='constant'
            )
        
        return fig

# Initialize dashboard
dashboard = ClusterVisualizationDashboard()

# Create Dash app
app = Dash(__name__)

# Compact header and full-height layout so main area can fill viewport without scrolling
app.layout = html.Div([
    # Header row: title and controls
    html.Div([
        html.H1("🎯 Interactive Cluster Visualization Dashboard",
                style={'margin': '0', 'fontSize': '20px', 'flex': '0 0 auto'}),

        html.Div([
            html.Div([
                html.Label("🔍 Select Clustering Algorithm & Parameters:", 
                          style={'font-weight': 'bold', 'marginBottom': '6px'}),
                dcc.Dropdown(
                    id='clustering-dropdown',
                    options=[
                        {'label': f"{result['display_name']} (Silhouette: {result['metrics']['silhouette']:.3f})" if result['metrics']['silhouette'] else result['display_name'], 
                         'value': key}
                        for key, result in dashboard.available_results.items()
                    ],
                    value=list(dashboard.available_results.keys())[0] if dashboard.available_results else None,
                    style={'width': '380px', 'margin': '2px 0'},
                    placeholder="Select a clustering result..."
                )
            ], style={'display': 'inline-block', 'verticalAlign': 'middle', 'marginRight': '12px'}),

            html.Div([
                dcc.Dropdown(
                    id='plot-type-dropdown',
                    options=[
                        {'label': '📈 PCA 2D', 'value': 'pca_2d'},
                        {'label': '🎲 PCA 3D', 'value': 'pca_3d'},
                        {'label': '🎯 t-SNE 2D', 'value': 'tsne_2d'},
                        {'label': '🌐 t-SNE 3D', 'value': 'tsne_3d'}
                    ],
                    value='pca_2d',
                    style={'width': '140px', 'margin': '2px 0'}
                )
            ], style={'display': 'inline-block', 'verticalAlign': 'middle'})
        ], style={'display': 'flex', 'alignItems': 'center', 'gap': '8px', 'flex': '1 1 auto', 'justifyContent': 'flex-end'})

    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'padding': '8px 12px', 'backgroundColor': '#f8f9fa', 'borderRadius': '6px', 'gap': '12px', 'height': '50px', 'boxSizing': 'border-box'}),

    # Info panels (compact)
    html.Div([
        html.Div(id="clustering-info", style={
            'flex': '1 1 50%',
            'margin': '0 12px 0 0',
            'padding': '10px',
            'border': '2px solid #e3e3e3',
            'borderRadius': '10px',
            'backgroundColor': '#ffffff',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
            'overflow': 'auto',
            'maxheight': '160px',
            'boxSizing': 'border-box'
        }),
        html.Div(id="method-info", style={
            'flex': '1 1 50%',
            'margin': '0 0 0 12px',
            'padding': '10px',
            'border': '2px solid #e3e3e3',
            'borderRadius': '10px',
            'backgroundColor': '#f8f9fa',
            'overflow': 'auto',
            'mazheight': '160px',
            'boxSizing': 'border-box'
        })
    ], style={'display': 'flex', 'alignItems': 'stretch', 'margin': '8px 0'}),
    
    # Main content: graph (flexible) + image panel (fixed width). This fills remaining viewport height.
    html.Div([
        # Graph container: grow to fill remaining space
        html.Div([
            html.Button("⤢ Fullscreen", id="fullscreen-btn", n_clicks=0, style={'marginBottom':'8px'}),
            dcc.Graph(id="main-graph", clear_on_unhover=True, style={'width': '100%', 'height': '100%'}),
            dcc.Tooltip(id="graph-tooltip")
        ], id='graph-container', style={'flex': '1 1 0%', 'boxSizing': 'border-box', 'padding': '8px', 'minWidth': '320px', 'height': '100%','overflow': 'hidden'}),
        
        # Image panel: fixed width, scrollable internally
        html.Div([
            dcc.Store(id='image-page', data=0),
            html.Label("🖼️ Show images for cluster:", style={'fontWeight': 'bold', 'marginBottom': '8px'}),
            dcc.Dropdown(id='cluster-select', options=[], value=None, style={'width': '100%', 'marginBottom': '8px'}),
            html.Div([
                html.Button('◀ Prev', id='prev-btn', n_clicks=0, style={'marginRight': '8px'}),
                html.Span(id='page-indicator', children='Page 1', style={'marginRight': '8px'}),
                html.Button('Next ▶', id='next-btn', n_clicks=0)
            ], style={'marginBottom': '8px', 'display': 'flex', 'alignItems': 'center'}),
            html.Div(id='image-grid', children=[], style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(120px, 1fr))', 'gap': '8px', 'width': '100%', 'justifyItems': 'center', 'alignItems': 'start', 'flex': '1 1 auto', 'overflow': 'hidden', 'gridAutoRows': 'auto'})
        ], id='image-panel', style={'flex': '0 0 360px', 'boxSizing': 'border-box', 'padding': '8px', 'minWidth': '280px', 'maxWidth': '420px', 'height': '100%', 'overflow': 'hidden', 'display': 'flex', 'flexDirection': 'column', 'backgroundColor': '#ffffff', 'borderLeft': '1px solid #e6e6e6'}),
    ], style={'display': 'flex', 'flex': '1 1 0%', 'alignItems': 'stretch', 'overflow': 'hidden', 'gap': '8px'}),
], style={'display': 'flex', 'flexDirection': 'column', 'height': '100vh', 'overflow': 'hidden', 'fontFamily': 'Arial, sans-serif', 'padding': '6px'})

@callback(
    [Output("main-graph", "figure"),
     Output("clustering-info", "children")],
    [Input("clustering-dropdown", "value"),
     Input("plot-type-dropdown", "value")]
)
def update_graph_and_info(clustering_key, plot_type):
    if clustering_key is None or not dashboard.available_results:
        empty_fig = go.Figure().add_annotation(
            text="No clustering results available.<br>Please run cluster analysis first.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return empty_fig, html.Div("❌ No clustering results available")
    
    # Load visualization data for selected clustering
    viz_data = dashboard.load_visualization_data(clustering_key)
    if viz_data is None:
        empty_fig = go.Figure().add_annotation(
            text="Visualization data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return empty_fig, html.Div("❌ Visualization data not available")
    
    # Update dashboard state
    # Ensure viz_data carries the experiment key so other callbacks can reference it
    if 'experiment_name' not in viz_data or not viz_data.get('experiment_name'):
        try:
            viz_data['experiment_name'] = clustering_key
        except Exception:
            pass
    dashboard.current_viz_data = viz_data
    
    # Get embeddings based on plot type
    embedding_key = f'embeddings_{plot_type}'
    embeddings = viz_data.get(embedding_key)
    
    if embeddings is None:
        empty_fig = go.Figure().add_annotation(
            text=f"No {plot_type} data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return empty_fig, html.Div(f"❌ No {plot_type} data available")
    
    clusters = viz_data['clusters']
    is_noise = viz_data.get('is_noise', np.zeros(len(clusters), dtype=bool))
    
    # Create title
    title = f"{plot_type.upper()} - {viz_data.get('experiment_name', clustering_key)}"
    is_3d = '3d' in plot_type
    
    # Create plot
    fig = dashboard.create_scatter_plot(embeddings, clusters, is_noise, title, is_3d)
    
    # Create clustering info panel
    result_info = dashboard.available_results[clustering_key]
    metrics = result_info['metrics']
    
    info_children = [
        html.H3(f"📊 {result_info['algorithm']} Results", 
                style={'color': '#2c3e50', 'marginBottom': '15px'}),
        
        # Render parameters and metrics side-by-side for a compact display
        html.Div([
            html.Div([
                html.Div([
                    html.H5("Parameters:", style={'color': '#34495e', 'marginBottom': '5px'}),
                    html.P(str(result_info['params']), style={'fontSize': '14px', 'margin': '0', 'wordBreak': 'break-all'})
                ], style={'minWidth': '220px', 'marginRight': '20px'}) ,

                html.Div([
                    html.H5("Clustering Metrics:", style={'color': '#34495e', 'marginBottom': '5px'}),
                    html.Div([
                        html.Span(f"🎯 Clusters: {metrics.get('n_clusters', 'N/A')}", 
                                 style={'fontSize': '14px'}),
                        html.Span(f"📈 Silhouette: {metrics.get('silhouette', 'N/A'):.3f}" if metrics.get('silhouette') else "📈 Silhouette: N/A", 
                                 style={'fontSize': '14px'}),
                        html.Span(f"🔇 Noise: {metrics.get('n_noise', 0)}" if result_info['algorithm'] == 'DBSCAN' else "", 
                                 style={'fontSize': '14px'})
                    ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center', 'gap': '12px', 'flexWrap': 'wrap'})
                ], style={'flex': '1', 'minWidth': '200px'})
            ], style={'display': 'flex', 'flexDirection': 'row', 'alignItems': 'flex-start', 'justifyContent': 'space-between', 'width': '100%'}),
        ])
    ]
    
    # Add additional metrics if available
    if metrics.get('calinski_harabasz'):
        info_children.append(
            html.P(f"📊 Calinski-Harabasz: {metrics['calinski_harabasz']:.3f}", 
                  style={'fontSize': '14px', 'margin': '5px 0'})
        )
    
    if metrics.get('davies_bouldin'):
        info_children.append(
            html.P(f"📉 Davies-Bouldin: {metrics['davies_bouldin']:.3f}", 
                  style={'fontSize': '14px', 'margin': '5px 0'})
        )
    
    return fig, html.Div(info_children)

@callback(
    [Output("graph-tooltip", "show"),
     Output("graph-tooltip", "bbox"),
     Output("graph-tooltip", "children")],
    [Input("main-graph", "hoverData")]
)
def display_hover(hoverData):
    if hoverData is None or dashboard.current_viz_data is None:
        return False, no_update, no_update

    # Get point information
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    
    # Get original index from customdata (handles filtered points)
    original_idx = pt.get("customdata", pt.get("pointNumber", 0))
    
    # Get data for this point
    clusters = dashboard.current_viz_data['clusters']
    image_paths = dashboard.current_viz_data['image_paths']
    is_noise = dashboard.current_viz_data.get('is_noise', np.zeros(len(clusters), dtype=bool))
    
    if original_idx >= len(clusters):
        return False, no_update, no_update
    
    cluster_id = clusters[original_idx]
    image_path = image_paths[original_idx]
    is_noise_point = is_noise[original_idx] if len(is_noise) > original_idx else False
    
    # Encode image for display
    img_src = dashboard.encode_image(image_path)
    
    # Create tooltip content
    cluster_text = "🔇 Noise" if cluster_id == -1 else f"🎯 Cluster {cluster_id}"
    
    children = [
        html.Div([
            html.Img(src=img_src, 
                    style={"width": "150px", "height": "150px", "object-fit": "cover", "borderRadius": "8px"}) 
                    if img_src else html.P("🖼️ Image not available", style={'color': '#7f8c8d'}),
            html.H4(cluster_text, 
                   style={"color": "#95a5a6" if cluster_id == -1 else "#3498db", "margin": "10px 0"}),
            html.P(f"📍 Point: {original_idx}", style={"margin": "5px 0", "fontSize": "12px"}),
            html.P(f"📁 {os.path.basename(image_path)}", 
                  style={"font-size": "11px", "margin": "5px 0", "color": "#7f8c8d"}),
        ], style={'width': '200px', 'white-space': 'normal', 'text-align': 'center', 'padding': '10px'})
    ]
    
    return True, bbox, children

@callback(
    Output("method-info", "children"),
    [Input("plot-type-dropdown", "value")]
)
def update_method_info(plot_type):
    if dashboard.current_viz_data is None:
        return html.Div()
    
    if plot_type.startswith('pca'):
        explained_var_key = f'{plot_type}_explained_variance'
        explained_var = dashboard.current_viz_data.get(explained_var_key, [])
        
        return html.Div([
            html.H4("📈 PCA Information", style={'color': '#2c3e50'}),
            html.P(f"📊 Explained variance per component: {[f'{x:.3f}' for x in explained_var]}"),
            html.P(f"🎯 Total explained variance: {sum(explained_var):.3f}", 
                  style={'fontWeight': 'bold'}),
            html.P("ℹ️ PCA preserves global structure and is good for understanding overall data distribution.")
        ])
    else:
        return html.Div([
            html.H4("🎯 t-SNE Information", style={'color': '#2c3e50'}),
            html.P("🔄 t-SNE is a non-linear dimensionality reduction technique"),
            html.P("🎯 Excellent for visualizing local cluster structure"),
            html.P("⚠️ Note: Doesn't preserve global distances between clusters"),
            html.P("🔇 Noise points (if any) may be excluded for better clarity", 
                  style={'fontStyle': 'italic', 'color': '#7f8c8d'})
        ])

@callback(
    [Output("graph-container", "style"), Output("fullscreen-btn", "children"), Output("main-graph", "style"), Output("image-panel", "style")],
    [Input("fullscreen-btn", "n_clicks")]
)
def toggle_fullscreen(n_clicks):
    """Toggle the main graph between normal and fullscreen modes using the button.
    Uses the parity of n_clicks to switch state.
    """
    # Default (normal) styles - keep a consistent half-screen split using flex basis so layout doesn't collapse
    # Normal (non-fullscreen) should fill the available container height
    normal_container_style = {'flex': '1 1 0%', 'boxSizing': 'border-box', 'margin': '0', 'minWidth': '320px', 'height': '100%'}
    normal_button_text = '⤢ Fullscreen'
    normal_graph_style = {'width': '100%', 'height': '90%'}

    # Fullscreen styles
    # Keep the graph fixed to the left half of the screen when toggled to fullscreen.
    fullscreen_container_style = {
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'width': '50%',
        'height': '100%',
        'zIndex': '9999',
        'backgroundColor': '#ffffff',
        'padding': '20px',
        'overflow': 'auto',
        'boxSizing': 'border-box'
    }
    fullscreen_button_text = '⤡ Exit Fullscreen'
    fullscreen_graph_style = {'width': '100%', 'height': 'calc(100vh - 80px)'}
    
    # Image panel styles to accompany fullscreen/normal modes. When graph is fixed to left half,
    # keep the image panel fixed to the right half so it remains visible and clipped.
    normal_image_panel_style = {'flex': '0 0 30%', 'boxSizing': 'border-box', 'margin': '0', 'overflow': 'hidden', 'minWidth': '320px', 'height': '100%'}
    fullscreen_image_panel_style = {
        'position': 'fixed',
        'top': '0',
        'right': '0',
        'width': '50%',
        'height': '100%',
        'zIndex': '9998',
        'overflow': 'auto',
        'backgroundColor': '#ffffff',
        'padding': '20px',
        'boxSizing': 'border-box'
    }

    if not n_clicks or (n_clicks % 2 == 0):
        return normal_container_style, normal_button_text, normal_graph_style, normal_image_panel_style
    else:
        return fullscreen_container_style, fullscreen_button_text, fullscreen_graph_style, fullscreen_image_panel_style

# New callbacks to populate cluster dropdown and show images (uses visualization data image paths)
@callback(
    [Output('cluster-select', 'options'), Output('cluster-select', 'value')],
    [Input('clustering-dropdown', 'value')]
)
def update_cluster_options_and_images(clustering_key):
    """Populate cluster options and initial image grid when clustering selection changes."""
    if clustering_key is None:
        return [], None

    # Prefer the already-loaded visualization data if it matches, otherwise load from disk
    viz_data = None
    if dashboard.current_viz_data is not None:
        # Try to detect matching experiment by file path or experiment_name
        try:
            if dashboard.current_viz_data.get('experiment_name') == clustering_key:
                viz_data = dashboard.current_viz_data
        except Exception:
            viz_data = None

    if viz_data is None:
        viz_data = dashboard.load_visualization_data(clustering_key)
    if viz_data is None:
        return [], None

    clusters = np.array(viz_data.get('clusters', []))
    image_paths = list(viz_data.get('image_paths', []))

    if clusters.size == 0:
        return [], None

    unique = np.unique(clusters)
    options = []
    for c in unique:
        label = 'Noise (-1)' if int(c) == -1 else f'Cluster {int(c)}'
        options.append({'label': label, 'value': int(c)})

    # default to first non-noise cluster if possible, else first
    default = None
    non_noise = [int(x) for x in unique if int(x) != -1]
    if non_noise:
        default = non_noise[0]
    else:
        default = int(unique[0])

    return options, default

@callback(
    [Output('image-grid', 'children'), Output('image-page', 'data'), Output('page-indicator', 'children')],
    [Input('prev-btn', 'n_clicks'), Input('next-btn', 'n_clicks'), Input('cluster-select', 'value'), Input('clustering-dropdown', 'value')],
    [State('image-page', 'data')]
)
def update_image_grid_and_page(prev_clicks, next_clicks, cluster_value, clustering_key, current_page):
    """Manage pagination and update image grid, page store, and page indicator in a single callback.
    This avoids race conditions between separate callbacks.
    """
    # Validate
    if clustering_key is None or cluster_value is None:
        return [], 0, 'Page 0 of 0'
    
    viz_data = dashboard.load_visualization_data(clustering_key)
    if viz_data is None:
        return [], 0, 'Page 0 of 0'
    
    clusters = np.array(viz_data.get('clusters', []))
    image_paths = list(viz_data.get('image_paths', []))

    if clusters.size == 0:
        return [], 0, 'Page 0 of 0'

    cluster_indices = np.where(clusters == cluster_value)[0]
    total = len(cluster_indices)
    per_page = 4
    max_pages = (total + per_page - 1) // per_page if total > 0 else 0

    # Determine trigger
    triggered = dash.callback_context.triggered
    trig_id = triggered[0]['prop_id'].split('.')[0] if triggered else None

    # Determine current page
    page = int(current_page) if current_page is not None else 0
    if trig_id in ('cluster-select', 'clustering-dropdown'):
        page = 0
    elif trig_id == 'next-btn':
        if page < max_pages - 1:
            page += 1
    elif trig_id == 'prev-btn':
        if page > 0:
            page -= 1

    # Clamp
    if page < 0:
        page = 0
    if max_pages > 0 and page > max_pages - 1:
        page = max_pages - 1

    # Build children for this page
    start = page * per_page
    end = start + per_page
    indices = cluster_indices[start:end]

    children = []
    for idx in indices:
        img_path = image_paths[idx] if idx < len(image_paths) else None
        img_src = dashboard.encode_image(img_path) if img_path else None
        if img_src:
            # compact centered 1:1 thumbnail (responsive: max size but will shrink to fit)
            tile = html.Div([
                html.Img(src=img_src, style={
                    'maxWidth': '400px',
                    'maxHeight': '400px',
                    'width': '100%',
                    'height': 'auto',
                    'objectFit': 'cover',
                    'borderRadius': '6px'
                }),
                html.Div(os.path.basename(img_path) if img_path else 'No image', style={
                    'textAlign': 'center',
                    'fontSize': '12px',
                    'marginTop': '6px',
                    'wordBreak': 'break-all',
                    'maxWidth': '100%'
                })
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'flex-start', 'width': '100%', 'maxWidth': '400px', 'boxSizing': 'border-box', 'gap':'6px'})
            children.append(tile)
        else:
            placeholder = html.Div([
                html.Div('🖼️', style={
                    'position': 'absolute',
                    'top': '50%',
                    'left': '50%',
                    'transform': 'translate(-50%, -50%)',
                    'fontSize': '24px',
                    'color': '#7f8c8d'
                })
            ], style={
                'position': 'relative',
                'width': '100%',
                'paddingTop': '100%',
                'overflow': 'hidden',
                'backgroundColor': '#ecf0f1',
                'borderRadius': '6px',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center',
                'color': '#7f8c8d',
                'fontSize': '24px',
                'maxWidth': '400px',
                'boxSizing': 'border-box'
            })
            wrapped = html.Div([placeholder, html.Div(os.path.basename(img_path) if img_path else 'No image', style={
                'textAlign': 'center', 'fontSize': '12px', 'marginTop': '6px', 'wordBreak': 'break-all', 'maxWidth': '100%'
            })], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '100%', 'maxWidth': '400px', 'boxSizing': 'border-box','gap':'6px'})
            children.append(wrapped)

    display = f'Page {page + 1} of {max_pages}' if max_pages > 0 else 'Page 0 of 0'
    return children, page, display

if __name__ == "__main__":
    if not dashboard.available_results:
        print("❌ No clustering results found!")
        print("📋 Please run the following commands:")
        print("   1. python cluster_analysis.py")
        print("   2. python prepare_visualization_data.py --all")
        print("   3. python dashboard.py")
    else:
        print(f"🎉 Found {len(dashboard.available_results)} clustering results")
        print("🚀 Starting dashboard...")
        app.run(debug=True)