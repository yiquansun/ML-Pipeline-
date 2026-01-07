import requests
import json

# 1. Update this with your actual Render URL
# Example: "https://ml-pipeline-jmr6.onrender.com/predict"
live_url = "https://ml-pipeline-jmr6.onrender.com/predict"

# 2. Data sample (Ensure it matches your Pydantic schema in main.py)
data = {
    "age": 32,
    "workclass": "Private",
    "fnlgt": 205019,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Sales",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States"
}

print(f"Sending POST request to: {live_url}")

try:
    # 3. Send the request
    response = requests.post(live_url, data=json.dumps(data))

    # 4. Print results for your Udacity screenshot
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(response.json())

except Exception as e:
    print(f"Error occurred: {e}")
