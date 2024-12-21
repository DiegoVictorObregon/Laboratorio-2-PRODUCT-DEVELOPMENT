import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import os

def load_pipeline(base_pipeline_path):
    # Cargar el pipeline base
    return joblib.load(base_pipeline_path)

def configure_and_train_pipeline(pipeline, X_train, y_train):
    # Agregar el modelo al pipeline
    pipeline.steps.append(('model', RandomForestClassifier()))
    # Entrenar el pipeline
    pipeline.fit(X_train, y_train)
    return pipeline

def save_pipeline(pipeline, path):
    # Guardar el pipeline entrenado
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)

if __name__ == "__main__":
    # Simulando datos de entrenamiento (reemplaza con tus datos reales)
    from sklearn.datasets import load_iris
    data = load_iris()
    X_train, y_train = data.data, data.target

    # Rutas de entrada y salida
    base_pipeline_path = "artifacts/base_pipeline.pkl"
    trained_pipeline_path = "artifacts/trained_pipeline.pkl"

    # Cargar, configurar y entrenar el pipeline
    pipeline = load_pipeline(base_pipeline_path)
    trained_pipeline = configure_and_train_pipeline(pipeline, X_train, y_train)
    save_pipeline(trained_pipeline, trained_pipeline_path)

    print("Pipeline entrenado guardado en artifacts/trained_pipeline.pkl")
