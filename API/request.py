import requests
import numpy as np

data = {
  "user_email": "moradi.b.k@gmail.com",
  "prob_name": "RealWorld",
  "prob_dim": 10,
  "num_sample_points": 2500,
  "doe": np.random.uniform(-5, 5, size=(2500, 11)).tolist(),
  "lb": (-5*np.ones(10)).tolist(),
  "ub": (+5*np.ones(10)).tolist()
}

response = requests.post(
    "http://localhost:8000/prediction",
    json=data,
    timeout=20
)
 
print(response)