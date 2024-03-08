from flask import Flask, request, jsonify
from chatgpt import chat_with_prompt

app = Flask(__name__)
chat_history = []

@app.route('/api/chat', methods=['POST'])
def chat():
    # Obtenemos el prompt del cuerpo de la solicitud
    data = request.get_json()
    prompt = data.get('prompt')

    # Llamamos a la función para chatear y obtener la respuesta
    response = chat_with_prompt(prompt, chat_history)

    # Devolvemos la respuesta en formato JSON
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
