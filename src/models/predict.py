import pandas as pd
import joblib
import os

def load_model():
    # Cargar el modelo entrenado
    model = joblib.load('models/random_forest_model.pkl')
    return model

def load_test_data():
    # Cargar los datos de prueba procesados
    test_data = pd.read_pickle('data/processed/test_featured.pkl')
    return test_data

def make_predictions(test_data, model):
    # Generar predicciones
    predictions = model.predict(test_data)
    return predictions

def save_predictions(predictions):
    # Crear el directorio si no existe
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar las predicciones en un archivo
    output_path = os.path.join(output_dir, 'test_predictions.csv')
    pd.DataFrame(predictions, columns=['Survived']).to_csv(output_path, index=False)
    print(f"Predicciones guardadas en {output_path}")

if __name__ == "__main__":
    model = load_model()
    test_data = load_test_data()
    predictions = make_predictions(test_data, model)
    save_predictions(predictions)

