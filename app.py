from flask import Flask
from fastai import *
from fastai.vision.all import *

app = Flask(__name__)

# Set path
path = Path()
path.ls(file_exts='.pkl')
path

# Load model
model = load_learner('./model.pkl')

@app.route('/')
def index():
    return 'hello, world'

@app.route('/classify', methods=['POST'])
def predict():
    prediction = model.predict('m_example.jpg')
    response = {
        "prediction": prediction[0]
    }
    return response

if __name__ == '__main__':
    app.run()