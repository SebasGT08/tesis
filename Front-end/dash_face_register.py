import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import base64

# Configuración de la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = 'Registro Facial'

# Diseño de la aplicación
app.layout = dbc.Container(
    [
        html.H1("Registro Facial", style={'textAlign': 'center', 'color': 'white'}),
        
        # Subida de imagen
        dcc.Upload(
            id='upload-image',
            children=html.Div([
                'Arrastra y suelta o ',
                html.A('Selecciona una Imagen', style={'color': '#2CA8FF'})
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px',
                'color': 'white'
            },
            multiple=False
        ),

        # Campo para el nombre
        dbc.Input(
            id="input-name",
            placeholder="Ingresa tu nombre",
            type="text",
            style={'margin': '10px'}
        ),

        # Botón para enviar datos
        dbc.Button(
            "Guardar",
            id="save-button",
            color="primary",
            style={'margin': '10px'}
        ),

        # Mostrar imágenes y mensajes
        dbc.Row(
            [
                dbc.Col(html.Div(id='original-image'), width=6),
                dbc.Col(html.Div(id='processed-image'), width=6),
            ]
        ),
        html.Div(id='output-message', style={'color': 'white', 'margin': '10px'}),
    ],
    fluid=True,
    style={'backgroundColor': '#2C2C2C'}
)

# Callback para mostrar las imágenes cargadas
@app.callback(
    [Output('original-image', 'children'), Output('processed-image', 'children')],
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def display_images(content, filename):
    if content:
        original_image = html.Img(src=content, style={'width': '100%', 'marginTop': '10px'})
        return original_image, None  # No procesamos la imagen en el frontend
    return None, None

# Callback para enviar datos al backend
@app.callback(
    Output('output-message', 'children'),
    [Input('save-button', 'n_clicks')],
    [State('upload-image', 'contents'), State('input-name', 'value')]
)
def enviar_datos_backend(n_clicks, image_content, name):
    if n_clicks:
        if image_content and name:
            # Extraer la parte base64 de la imagen
            image_data = image_content.split(",")[1]

            # Crear payload para el backend
            payload = {
                'name': name,
                'image': image_data
            }

            try:
                # Enviar datos al backend
                response = requests.post('http://127.0.0.1:5000/register', json=payload)

                # Evaluar la respuesta del backend
                if response.status_code == 200:
                    return "Registro exitoso."
                else:
                    return f"Error del backend: {response.json().get('error', 'Error desconocido')}"
            except Exception as e:
                return f"Error de conexión con el backend: {e}"
        else:
            return "Por favor, carga una imagen y escribe un nombre."
    return ""

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
