from flask import Flask, request, jsonify
from flask_cors import CORS
from chatgpt import chat_with_prompt

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, allow_headers="*")

chat_history = []

@app.route('/api/chat', methods=['POST'])
def chat():
    # Obtenemos el prompt del cuerpo de la solicitud
    data = request.get_json()
    prompt = data.get('prompt')

    # Llamamos a la función para chatear y obtener la respuesta
    response = chat_with_prompt(prompt, chat_history)
    chat_history.append((prompt, response))

    # Devolvemos la respuesta en formato JSON
    return jsonify({'response': response})

# Esta parte solo se ejecutará si ejecutas este script directamente, no con Gunicorn
if __name__ == '__main__':
    # En lugar de usar app.run(), ejecutaremos la aplicación con Gunicorn
    # Especificamos el número de workers y la dirección donde Gunicorn debe escuchar las solicitudes
    # En este caso, escucharemos en el puerto 5000 en la dirección local
    app.run(host='127.0.0.1', port=5000)
 