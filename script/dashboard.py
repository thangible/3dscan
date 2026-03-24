import numpy as np
import base64
import io
import os
import json
import glob
from PIL import Image
from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
import plotly.express as px

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
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    zaxis_title="Component 3"
                ),
                height=600
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
            
            fig.update_layout(
                title=title,
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                height=500
            )
        
        return fig

# Initialize dashboard
dashboard = ClusterVisualizationDashboard()

# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("🎯 Interactive Cluster Visualization Dashboard", 
            style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    html.Div([
        html.Div([
            html.Label("🔍 Select Clustering Algorithm & Parameters:", 
                      style={'font-weight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='clustering-dropdown',
                options=[
                    {'label': f"{result['display_name']} (Silhouette: {result['metrics']['silhouette']:.3f})" if result['metrics']['silhouette'] else result['display_name'], 
                     'value': key}
                    for key, result in dashboard.available_results.items()
                ],
                value=list(dashboard.available_results.keys())[0] if dashboard.available_results else None,
                style={'width': '500px', 'margin': '10px 0'},
                placeholder="Select a clustering result..."
            )
        ], style={'display': 'inline-block', 'margin-right': '30px', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("📊 Select Visualization Type:", 
                      style={'font-weight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='plot-type-dropdown',
                options=[
                    {'label': '📈 PCA 2D', 'value': 'pca_2d'},
                    {'label': '🎲 PCA 3D', 'value': 'pca_3d'},
                    {'label': '🎯 t-SNE 2D', 'value': 'tsne_2d'},
                    {'label': '🌐 t-SNE 3D', 'value': 'tsne_3d'}
                ],
                value='pca_2d',
                style={'width': '200px', 'margin': '10px 0'}
            )
        ], style={'display': 'inline-block', 'verticalAlign': 'top'})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # Clustering info panel
    html.Div(id="clustering-info", style={
        'margin': '20px', 
        'padding': '20px', 
        'border': '2px solid #e3e3e3', 
        'borderRadius': '10px',
        'backgroundColor': '#ffffff',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Main graph with fullscreen support
    html.Div([
        html.Div([
            html.Button("⤢ Fullscreen", id="fullscreen-btn", n_clicks=0, style={'marginBottom':'10px'}),
            dcc.Graph(id="main-graph", clear_on_unhover=True, style={'width': '100%', 'height': '500px'}),
            dcc.Tooltip(id="graph-tooltip")
        ], id='graph-container', style={'margin': '20px'})
    ]),
    
    # Method info panel
    html.Div(id="method-info", style={
        'margin': '20px', 
        'padding': '20px', 
        'border': '2px solid #e3e3e3', 
        'borderRadius': '10px',
        'backgroundColor': '#f8f9fa'
    })
], style={'fontFamily': 'Arial, sans-serif'})

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
        
        html.Div([
            html.Div([
                html.H5("Parameters:", style={'color': '#34495e', 'marginBottom': '5px'}),
                html.P(str(result_info['params']), style={'fontSize': '14px', 'margin': '0'})
            ], style={'marginBottom': '15px'}),
            
            html.Div([
                html.H5("Clustering Metrics:", style={'color': '#34495e', 'marginBottom': '10px'}),
                html.Div([
                    html.Span(f"🎯 Clusters: {metrics.get('n_clusters', 'N/A')}", 
                             style={'marginRight': '20px', 'fontSize': '14px'}),
                    html.Span(f"📈 Silhouette: {metrics.get('silhouette', 'N/A'):.3f}" if metrics.get('silhouette') else "📈 Silhouette: N/A", 
                             style={'marginRight': '20px', 'fontSize': '14px'}),
                    html.Span(f"🔇 Noise: {metrics.get('n_noise', 0)}" if result_info['algorithm'] == 'DBSCAN' else "", 
                             style={'fontSize': '14px'})
                ])
            ])
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
    [Output("graph-container", "style"), Output("fullscreen-btn", "children"), Output("main-graph", "style")],
    [Input("fullscreen-btn", "n_clicks")]
)
def toggle_fullscreen(n_clicks):
    """Toggle the main graph between normal and fullscreen modes using the button.
    Uses the parity of n_clicks to switch state.
    """
    # Default (normal) style
    normal_container_style = {'margin': '20px'}
    normal_button_text = '⤢ Fullscreen'
    normal_graph_style = {'width': '100%', 'height': '500px'}

    # Fullscreen styles
    fullscreen_container_style = {
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'width': '100%',
        'height': '100%',
        'zIndex': '9999',
        'backgroundColor': '#ffffff',
        'padding': '20px',
        'overflow': 'auto'
    }
    fullscreen_button_text = '⤡ Exit Fullscreen'
    fullscreen_graph_style = {'width': '100%', 'height': 'calc(100vh - 80px)'}

    if not n_clicks or (n_clicks % 2 == 0):
        return normal_container_style, normal_button_text, normal_graph_style
    else:
        return fullscreen_container_style, fullscreen_button_text, fullscreen_graph_style

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