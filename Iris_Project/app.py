from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model you saved earlier
# Ensure the file 'knn_model.joblib' is in this same folder!
import os

# Get the directory where app.py is located
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'knn_model.joblib')

# Load the model using the absolute path
model = joblib.load(model_path)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from the HTML form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        
        # Map numeric result (0, 1, 2) to class name
        species_names = ['Setosa', 'Versicolor', 'Virginica']
        output = species_names[prediction[0]]

        return render_template('index.html', prediction_text=f'Predicted Species: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)