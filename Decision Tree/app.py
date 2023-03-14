import pickle
from flask import Flask, request, app, render_template
import pandas as pd

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home_page():
    return render_template('index.html')


@app.route("/predict", methods = ["POST"])
def prediction():
    if (request.method=='POST'):
        input_data=request.form['input_data']
        input_data_updated = list(map(float, input_data.split(', ')))
        with open('model.pkl' , 'rb') as f:
            lr = pickle.load(f)
            result = lr.predict([input_data_updated])
        return render_template('results.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)