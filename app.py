from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Validate input ranges
        if not (4.3 <= sepal_length <= 7.9):
            return render_template('index.html', error='Sepal Length must be between 4.3 and 7.9 cm')
        if not (2.0 <= sepal_width <= 4.4):
            return render_template('index.html', error='Sepal Width must be between 2.0 and 4.4 cm')
        if not (1.0 <= petal_length <= 6.9):
            return render_template('index.html', error='Petal Length must be between 1.0 and 6.9 cm')
        if not (0.1 <= petal_width <= 2.5):
            return render_template('index.html', error='Petal Width must be between 0.1 and 2.5 cm')

        # Scale features
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        return render_template('index.html', 
                             prediction=prediction,
                             sepal_length=sepal_length,
                             sepal_width=sepal_width,
                             petal_length=petal_length,
                             petal_width=petal_width)

    except ValueError:
        return render_template('index.html', error='Please enter valid numbers for all measurements')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 