
from API.services.get_production_model import get_production_model
import joblib

def predict(features, conn):
    
    production_model = get_production_model()
    
    with open(f"{production_model['path']}", "rb") as f:
        model = joblib.load(f)
        
    predicted_labels = model.predict(features)[0]
    
    with conn.cursor() as cur:
        cur.execute("SELECT algorithm_name FROM algorithms ORDER BY algorithm_id")
        algorithm_list = [row[0] for row in cur.fetchall()]

    predicted_algorithms = [algorithm for algorithm, label in zip(algorithm_list, predicted_labels) if label == 1]
    
    return predicted_algorithms