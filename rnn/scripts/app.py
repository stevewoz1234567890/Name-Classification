from flask import Flask, request, jsonify
from pathlib import Path
from predict import predict

# Initialize Flask app
app = Flask(__name__)

# Define the REST API endpoint
@app.route('/classify', methods=['POST'])
def classify_name():
    # Extract the JSON data sent to the endpoint
    data = request.json
    
    # Get the name to classify and the number of predictions (n_predictions)
    name = data.get('name')
    n_predictions = data.get('n_predictions', 3)  # default to top 3 predictions
    
    # Path to the model weights (you can adjust this path if needed)
    weights_file = Path('./weights.pt')

    # Run the prediction
    try:
        predictions = predict(name=name, n_predictions=n_predictions, weights_file=weights_file)
        
        # Return the predictions as a JSON response
        return jsonify({
            'name': name,
            'predictions': [
                {'label': label, 'score': value.item()} for value, label in predictions
            ]
        }), 200
    except Exception as e:
        # If there's an error, return the error message
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
