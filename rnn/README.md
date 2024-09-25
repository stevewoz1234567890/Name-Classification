# Name classification API
## Task description
You are provided a set of scripts:
- `scripts/train.py` does model training
- `scripts/predict.py` does name classification based on the trained model

Your task is:
1. Expose the name classification via REST api endpoint. The endpoint should provide the ability to select top N most likely labels for the given name and should also provide the scores associated with each label.
2. Containerize the said API
3. Deploy the said container to k8s cluster using helm chart
4. Provide a document (readme) describing how to deploy and use the API.

You are free to use any REST API framework or library and design the endpoint as you see fit.
You are free to duplicate and edit the code from `scripts/` folder in a way you see would work best, as long as the classification can be run using your API.
You are encouraged to use a local distribution of k8s like `minikube`.

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

