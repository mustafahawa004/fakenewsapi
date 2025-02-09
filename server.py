from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Fake News Detection API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news_text = data.get("text", "")

    if not news_text.strip():
        return jsonify(["error:" "Empty text input!"]), 400
    
    prediction = model.predict([news_text])[0]
    result = "Fake News" if prediction == 1 else "Real News"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
