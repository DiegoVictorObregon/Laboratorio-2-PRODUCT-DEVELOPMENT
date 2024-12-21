import joblib
import pandas as pd
from datetime import datetime
import os

def load_pipeline(pipeline_path):
    # Cargar el pipeline entrenado
    return joblib.load(pipeline_path)

def predict_and_save(pipeline, X_test, output_dir):
    # Realizar predicciones
    predictions = pipeline.predict(X_test)
    # Generar un nombre único para el archivo de salida
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    os.makedirs(output_dir, exist_ok=True)
    # Guardar las predicciones en un archivo CSV
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_path, index=False)
    print(f"Predicciones guardadas en {output_path}")

if __name__ == "__main__":
    # Simulando datos de prueba (reemplaza con tus datos reales)
    from sklearn.datasets import load_iris
    data = load_iris()
    X_test = data.data

    # Ruta del pipeline entrenado y carpeta de salida
    trained_pipeline_path = "artifacts/trained_pipeline.pkl"
    output_dir = "data/predictions"

    # Cargar el pipeline entrenado y realizar predicciones
    pipeline = load_pipeline(trained_pipeline_path)
    predict_and_save(pipeline, X_test, output_dir)
