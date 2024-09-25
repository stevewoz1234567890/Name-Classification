# Name classification API
## Task description
You are provided a set of scripts:
- `scripts/train.py` does model training
- `scripts/predict.py` does name classification based on the trained model


## Setup
```
pip install -r requirements.txt
```
## Train the model
```
python scripts/train.py
```
This will save the weights file in the current directory. Training takes a few mins on CPU.
## Run inference via CLI
Example of running the script:
```
python scripts/predict.py Quang
```
Output will look like so
```
(-0.02) Vietnamese
(-4.66) Chinese
(-5.43) Korean
```

## Your task is:

### 1. Expose the name classification via REST api endpoint. The endpoint should provide the ability to select top N most likely labels for the given name and should also provide the scores associated with each label.

This is going to be a name classification model exposed via a REST API endpoint. Users can provide the API with a name and top N most likely categories (labels) with scores. Here, this includes the use of Flask and a PyTorch model.

#### Start the Flask Server
```
python app.py
```
This will start the Flask server on http://localhost:5000.

#### Request Structure
JSON payload:
    name: The name to classify.
    n_predictions: (optional) Number of top predictions to return. Default is 3.

Example request body:

```
{
  "name": "John",
  "n_predictions": 3
}
```

#### Response Structure

Example response:

```
{
  "name": "John",
  "predictions": [
    {"label": "English", "score": -0.95},
    {"label": "German", "score": -1.23},
    {"label": "Dutch", "score": -1.67}
  ]
}
```

label: The predicted category (e.g., nationality of the name).
score: The confidence score (lower is better; the model uses negative log likelihood).

#### Testing the API with Postman
URL: http://localhost:5000/classify
Headers: Set the Content-Type to application/json.
In the body, select raw and choose JSON, then provide the JSON payload:
```
{
  "name": "John",
  "n_predictions": 3
}
```

### 2. Containerize the said API

#### Build the Docker Image

```
docker build -t name-classification-api .
```

#### Run the Docker Container

```
docker run -p 5000:5000 name-classification-api
```

#### Test the API using Postman

Endpoint: http://localhost:5000/classify
Method: POST
Headers:
  Content-Type: application/json
Body (JSON):
```
{
  "name": "John",
  "n_predictions": 3
}
```
Response:
```
{
  "name": "John",
  "predictions": [
    {"label": "English", "score": -0.95},
    {"label": "German", "score": -1.23},
    {"label": "Dutch", "score": -1.67}
  ]
}
```

#### Stop the Docker Container

```
docker stop <container_id>
```

3. Deploy the said container to k8s cluster using helm chart
4. Provide a document (readme) describing how to deploy and use the API.

You are free to use any REST API framework or library and design the endpoint as you see fit.
You are free to duplicate and edit the code from `scripts/` folder in a way you see would work best, as long as the classification can be run using your API.
You are encouraged to use a local distribution of k8s like `minikube`.
