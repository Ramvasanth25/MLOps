import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pre-trained model from a pickle file
with open('employee.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the inputs from the form
    department = request.form.get('department')
    employee_id = request.form.get('employee_id')

    # Convert input to appropriate format for the model
    # Example: Map department to a numeric or one-hot encoding
    # Here, assuming 'department' is one-hot encoded during training
    department_mapping = {'HR': 0, 'Finance': 1, 'IT': 2}  # Adjust according to training
    department_encoded = department_mapping.get(department, -1)  # Default to -1 if invalid

    try:
        # Convert Employee ID to numeric
        employee_id = int(employee_id)
    except ValueError:
        return render_template('result.html', prediction="Invalid Employee ID format.")

    # Combine inputs into the model's expected format
    input_data = [employee_id, department_encoded]  # Adjust based on model input structure

    # Make a prediction
    try:
        prediction = model.predict([input_data])[0]  # Assumes the model's predict method works
    except Exception as e:
        return render_template('result.html', prediction=f"Error in prediction: {str(e)}")

    # Render the result.html with the prediction
    return render_template('result.html', prediction=f"The predicted output is: {prediction}")

if __name__ == '__main__':
    app.run(debug=True)
