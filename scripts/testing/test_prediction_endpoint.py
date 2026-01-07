import requests

# Define the URL for the prediction endpoint
url = "http://localhost:8000/predict"

# Create a valid input payload
payload = {
    "features": [0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
}

# Send a POST request to the prediction endpoint
response = requests.post(url, json=payload)

# Print the response
print("Status Code:", response.status_code)
print("Response:", response.json())
