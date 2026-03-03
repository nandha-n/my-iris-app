from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'knn_model.joblib')
# scaler_path = os.path.join(base_dir, 'scaler.joblib') # Uncomment if you used a scaler

# Load model
model = joblib.load(model_path)
# scaler = joblib.load(scaler_path) # Uncomment if you used a scaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Collect data explicitly by name to prevent ordering issues
        # These names must match the 'name' attribute in your HTML <input> tags
        input_data = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])
        ]
        
        # 2. Reshape for Scikit-Learn (expects 2D array)
        final_features = np.array(input_data).reshape(1, -1)
        
        # 3. Apply Scaling (CRITICAL: Only if you scaled during training!)
        # final_features = scaler.transform(final_features)
        
        # 4. Make prediction
        prediction = model.predict(final_features)
        
        # 5. Map numeric result to class name
        species_names = ['Setosa', 'Versicolor', 'Virginica']
        output = species_names[prediction[0]]

        return render_template('index.html', prediction_text=f'Result: {output}')

    except Exception as e:
        # Useful for debugging; in production, you might want a cleaner message
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
    import plotly.express as px
import plotly.io as pio
import pandas as pd
from sklearn.datasets import load_iris

# ... (your existing model loading code) ...

# Load the dataset for the background chart
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        f1 = float(request.form['sepal_length'])
        f2 = float(request.form['sepal_width'])
        f3 = float(request.form['petal_length'])
        f4 = float(request.form['petal_width'])
        
        features = np.array([[f1, f2, f3, f4]])
        prediction = model.predict(features)
        species_names = ['Setosa', 'Versicolor', 'Virginica']
        output = species_names[prediction[0]]

        # --- CREATE THE CHART ---
        fig = px.scatter(df, x="petal length (cm)", y="petal width (cm)", 
                         color="species", title="Your Input vs. Iris Dataset",
                         labels={"petal length (cm)": "Petal Length", "petal width (cm)": "Petal Width"})
        
        # Add the user's point as a big Red X
        fig.add_scatter(x=[f3], y=[f4], name="Your Input", 
                        marker=dict(color='red', size=15, symbol='x'))

        # Convert plot to HTML string
        graph_html = pio.to_html(fig, full_html=False)

        return render_template('index.html', 
                               prediction_text=f'Predicted: {output}', 
                               plot=graph_html)
    except Exception as e:
        return f"Error: {e}"

