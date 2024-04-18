import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# Define CSS styles
styles = {
    'dashboard': {
        'textAlign': 'center'
    },
    'dashboard-logo': {
        'height': '60px', 'marginRight': '15px'
    },
    'dashboard-header': {
        'backgroundColor': '#282c34',
        'minHeight': '10vh',
        'display': 'flex',
        'flexDirection': 'row',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'fontSize': '20px',
        'color': 'white',
        'paddingLeft': '10px',
        'borderBottom': '3px solid #61dafb'
    },
    'title-container': {
        'display': 'flex', 'flexDirection': 'row', 'alignItems': 'center'
    },
    'dashboard-title': {
        'color': '#61dafb', 'textTransform': 'uppercase', 'letterSpacing': '2px'
    },
    'dashboard-button': {
        'padding': '10px 20px',
        'fontSize': '20px',
        'color': '#fff',
        'backgroundColor': '#61dafb',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'marginRight': '20px',
        'transition': 'background-color 0.3s ease'
    },
    'dashboard-button:hover': {
        'backgroundColor': '#4098da'
    },
    'dashboard-layout': {
        'display': 'flex',
        'flexWrap': 'wrap',
        'justifyContent': 'space-around',
        'gridTemplateColumns': 'repeat(auto-fill, minmax(200px, 1fr))',
        'gap': '20px',
        'padding': '20px'
    },
    'dashboard-chart': {
        'width': '45%', 'height': '200px', 'border': '1px solid #ccc', 'margin': '10px'
    },
    'dashboard-table': {
        'width': '45%', 'height': '200px', 'border': '1px solid #ccc', 'margin': '10px'
    }
}

# App layout
app.layout = html.Div(
    style=styles['dashboard'],
    children=[
        html.Header(
            style=styles['dashboard-header'],
            children=[
                html.Div(
                    style=styles['title-container'],
                    children=[
                        html.Img(
                            style=styles['dashboard-logo'],
                            src="https://img.icons8.com/color/48/lungs.png",
                            alt="logo"
                        ),
                        html.H1(
                            style=styles['dashboard-title'],
                            children="Pneumonia Detection"
                        )
                    ]
                ),
                html.Button(
                    id='train-button',
                    style=styles['dashboard-button'],
                    children="Lancer l'entrainement du modèle",
                    n_clicks=0
                )
            ]
        ),
        html.Div(
            style=styles['dashboard-layout'],
            children=[
                dcc.Graph(id='graph-1', style=styles['dashboard-chart']),
                dcc.Graph(id='graph-2', style=styles['dashboard-chart']),
                html.Div(id='table', style=styles['dashboard-table']),
                dcc.Graph(id='graph-3', style=styles['dashboard-chart']),
            ]
        )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)