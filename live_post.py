import requests

# Once you deploy to Render/Heroku, replace this URL with your live URL
live_url = "https://your-app-name.onrender.com/predict"

data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

r = requests.post(live_url, json=data)

print(f"Status Code: {r.status_code}")
print(f"Prediction: {r.json()['prediction']}")