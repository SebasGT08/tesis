import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import socketio
import requests

# Configuración de la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
app.title = 'Registro Facial'

# Inicializar cliente de SocketIO
sio = socketio.Client()

# Variable para almacenar el último frame recibido
latest_frame = None

# Evento para recibir frames
@sio.on('video_frame')
def handle_video_frame(data):
    global latest_frame
    latest_frame = data['frame']  # Guardar el último frame recibido

# Conectar al servidor Flask-SocketIO
sio.connect('http://127.0.0.1:5000')  # Asegúrate de que el servidor Flask-SocketIO esté corriendo

# Diseño de la aplicación
app.layout = dbc.Container(
    [
        html.H1("Registro Facial", style={'textAlign': 'center', 'color': 'white'}),

        # Menú con pestañas
        dbc.Tabs(
            [
                dbc.Tab(label="Registros", tab_id="registros"),
                dbc.Tab(label="Cámaras", tab_id="camaras"),
            ],
            id="tabs",
            active_tab="registros",
            style={'marginBottom': '20px'}
        ),

        # Contenedor para el contenido de cada pestaña
        html.Div(id="tab-content", style={'backgroundColor': '#2C2C2C', 'height': '100%'}),
    ],
    fluid=True,
    style={'backgroundColor': '#2C2C2C', 'height': '100vh'}
)

# Callback para cambiar el contenido según la pestaña seleccionada
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "registros":
        return html.Div(
            [
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
            ]
        )
    elif active_tab == "camaras":
        return html.Div(
            [
                # Ventana animada para la cámara
                html.Div(
                    id="camera-window",
                    children=[
                        html.H4("Stream de Cámara", style={'color': 'white', 'textAlign': 'center'}),
                        html.Img(
                            id="video-feed",
                            style={'width': '100%', 'height': '100%', 'border': '2px solid white'}
                        ),
                    ],
                    style={
                        'position': 'relative',
                        'width': '50%',
                        'height': '50%',
                        'backgroundColor': '#2C2C2C',
                        'border': '1px solid white',
                        'borderRadius': '10px',
                        'overflow': 'hidden',
                        'zIndex': 2000
                    }
                ),
                # Intervalo para actualizar frames del stream
                dcc.Interval(id="interval", interval=50),  # Intervalo para actualizar frames (~20 FPS)
            ]
        )
    return "Seleccione una pestaña para ver el contenido."

# Callback para actualizar el stream de video
@app.callback(
    Output("video-feed", "src"),
    [Input("interval", "n_intervals"), Input("tabs", "active_tab")]
)
def update_frame(_, active_tab):
    if active_tab != "camaras":
        return dash.no_update
    global latest_frame
    if latest_frame:
        return f"data:image/jpeg;base64,{latest_frame}"
    return dash.no_update

# Callback para mostrar las imágenes cargadas
@app.callback(
    [Output('original-image', 'children'), Output('processed-image', 'children')],
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def display_images(content, filename):
    if content:
        original_image = html.Img(src=content, style={'width': '100%', 'marginTop': '10px'})
        return original_image, None
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
