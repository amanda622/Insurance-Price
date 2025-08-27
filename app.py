from flask import Flask, render_template, request
# To load the trained model
from joblib import load


# Load the trained model
loaded_convertor = load("poly_d3_conv_20250117.joblib")
loaded_model = load("model_d3_poly_20250117.joblib")

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # Render the HTML form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    age = int(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = 1 if request.form['smoker'] == 'yes' else 0

    # Prepare input for the model
    inputs = [[age, bmi, children, smoker]]
    sales_prediction = loaded_model.predict(loaded_convertor.fit_transform(inputs))[0]

    # Return the prediction to the user
    return render_template('result.html', prediction=f"${sales_prediction:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
