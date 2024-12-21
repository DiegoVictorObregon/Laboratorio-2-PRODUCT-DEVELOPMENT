from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

def create_pipeline():
    # Crear un pipeline base con un escalador estándar
    pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])
    return pipeline

def save_pipeline(pipeline, path):
    # Guardar el pipeline en la ruta especificada
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)

if __name__ == "__main__":
    pipeline = create_pipeline()
    save_pipeline(pipeline, "artifacts/base_pipeline.pkl")
    print("Pipeline base guardado en artifacts/base_pipeline.pkl")
