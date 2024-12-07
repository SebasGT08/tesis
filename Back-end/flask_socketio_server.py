# flask_socketio_server.py

import eventlet
eventlet.monkey_patch()

from flask import Flask
from flask_socketio import SocketIO, emit

# Inicializar Flask y SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def handle_connect():
    print("Un cliente se ha conectado.")

@socketio.on('disconnect')
def handle_disconnect():
    print("Un cliente se ha desconectado.")

@socketio.on('video_frame')
def handle_video_frame(data):
    frame = data.get('frame')
    if frame:
        # Emitir el frame a todos los clientes conectados excepto al remitente
        emit('video_frame', {'frame': frame}, broadcast=True, include_self=False)

if __name__ == '__main__':
    print("Servidor Flask con SocketIO iniciando en http://127.0.0.1:5000")
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
