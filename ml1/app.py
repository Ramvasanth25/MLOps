import pickle
from flask import Flask, request, render_template
from numpy import vectorize
from sklearn.calibration import LabelEncoder

app = Flask(__name__)

# Load the pre-trained model from a pickle file
with open('exp1.pkl', 'rb') as file:
    model = pickle.load(file)
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the form
    input_text = request.form.get('t1')  # Raw text from the form

    try:
        # Vectorize the input text
        input_vector = vectorizer.transform([input_text])  # Transform text to numerical features

        # Convert sparse matrix to dense if necessary
        input_data = input_vector.toarray()  # Ensure the input is a 2D array

        # Debugging: Print the shape of input_data
        print(f"Shape of input_data: {input_data.shape}")  # Should be (1, n_features)

        # Make a prediction
        prediction = model.predict(input_data)[0]  # Model expects a 2D array, input_data is now correct

        # Convert prediction to human-readable format
        sentiment = "Positive" if prediction == 1 else "Negative"

    except Exception as e:
        sentiment = f"Error during prediction: {e}"

    # Render the result
    return render_template('result.html', prediction=f"The sentiment is: {sentiment}")

if __name__ == '__main__':
    app.run(debug=True)
