import numpy as np
import base64
import io
from PIL import Image
from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
import plotly.express as px

# Load the visualization data
viz_data = np.load("exp/visualization_data.npy", allow_pickle=True).item()

embeddings_2d = viz_data['embeddings_pca_2d']
embeddings_3d = viz_data['embeddings_pca_3d']
embeddings_tsne_2d = viz_data['embeddings_tsne_2d']
embeddings_tsne_3d = viz_data['embeddings_tsne_3d']
clusters = viz_data['clusters']
image_paths = viz_data['image_paths']

def encode_image(image_path):
    """Convert image to base64 for display in tooltip"""
    try:
        with Image.open(image_path) as img:
            # Resize for faster loading
            img.thumbnail((200, 200))
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{encoded_image}"
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def create_scatter_plot(embeddings, title, is_3d=False):
    """Create 2D or 3D scatter plot"""
    # Create color map for clusters
    unique_clusters = np.unique(clusters)
    colors = px.colors.qualitative.Set3[:len(unique_clusters)]
    color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
    point_colors = [color_map[cluster] for cluster in clusters]
    
    if is_3d:
        fig = go.Figure(data=[go.Scatter3d(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            z=embeddings[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=point_colors,
                opacity=0.7,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            text=[f"Cluster {c}" for c in clusters],
            hovertemplate="<b>Cluster %{text}</b><br>" +
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
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=point_colors,
                opacity=0.7,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            text=[f"Cluster {c}" for c in clusters],
            hovertemplate="<b>Cluster %{text}</b><br>" +
                         "X: %{x:.2f}<br>" +
                         "Y: %{y:.2f}<extra></extra>"
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            height=500
        )
    
    # Disable default hover for custom tooltip
    fig.update_traces(hoverinfo="none", hovertemplate=None)
    return fig

# Create figures
fig_pca_2d = create_scatter_plot(embeddings_2d, "PCA 2D Visualization", is_3d=False)
fig_pca_3d = create_scatter_plot(embeddings_3d, "PCA 3D Visualization", is_3d=True)
fig_tsne_2d = create_scatter_plot(embeddings_tsne_2d, "t-SNE 2D Visualization", is_3d=False)
fig_tsne_3d = create_scatter_plot(embeddings_tsne_3d, "t-SNE 3D Visualization", is_3d=True)

# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Cluster Data Visualization", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Select Visualization Type:"),
        dcc.Dropdown(
            id='plot-type-dropdown',
            options=[
                {'label': 'PCA 2D', 'value': 'pca_2d'},
                {'label': 'PCA 3D', 'value': 'pca_3d'},
                {'label': 't-SNE 2D', 'value': 'tsne_2d'},
                {'label': 't-SNE 3D', 'value': 'tsne_3d'}
            ],
            value='pca_2d',
            style={'width': '200px', 'margin': '10px'}
        )
    ]),
    
    dcc.Graph(id="main-graph", clear_on_unhover=True),
    dcc.Tooltip(id="graph-tooltip"),
    
    html.Div(id="info-panel", style={'margin': '20px', 'padding': '10px', 'border': '1px solid #ccc'})
])

@callback(
    Output("main-graph", "figure"),
    Input("plot-type-dropdown", "value")
)
def update_graph(plot_type):
    if plot_type == 'pca_2d':
        return fig_pca_2d
    elif plot_type == 'pca_3d':
        return fig_pca_3d
    elif plot_type == 'tsne_2d':
        return fig_tsne_2d
    elif plot_type == 'tsne_3d':
        return fig_tsne_3d

@callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("main-graph", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    # Get point information
    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]
    
    # Get data for this point
    cluster_id = clusters[num]
    image_path = image_paths[num]
    
    # Encode image for display
    img_src = encode_image(image_path)
    
    # Create tooltip content
    children = [
        html.Div([
            html.Img(src=img_src, style={"width": "150px", "height": "150px", "object-fit": "cover"}) if img_src else html.P("Image not available"),
            html.H4(f"Cluster: {cluster_id}", style={"color": "darkblue"}),
            html.P(f"Point Index: {num}"),
            html.P(f"Image: {image_path.split('/')[-1]}", style={"font-size": "12px"}),
        ], style={'width': '200px', 'white-space': 'normal', 'text-align': 'center'})
    ]
    
    return True, bbox, children

@callback(
    Output("info-panel", "children"),
    Input("plot-type-dropdown", "value")
)
def update_info_panel(plot_type):
    if plot_type.startswith('pca'):
        explained_var = viz_data['pca_2d_explained_variance'] if '2d' in plot_type else viz_data['pca_3d_explained_variance']
        return html.Div([
            html.H4("PCA Information"),
            html.P(f"Explained variance ratio: {explained_var}"),
            html.P(f"Total explained variance: {sum(explained_var):.3f}")
        ])
    else:
        return html.Div([
            html.H4("t-SNE Information"),
            html.P("t-SNE is a non-linear dimensionality reduction technique"),
            html.P("Good for visualizing clusters but doesn't preserve global structure")
        ])

if __name__ == "__main__":
    app.run(debug=True)