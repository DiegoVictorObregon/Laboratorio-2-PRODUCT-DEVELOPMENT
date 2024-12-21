import pandas as pd

def load_data():
    train_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'
    
    # Cargando los datos de entrenamiento y prueba
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    return train_data, test_data

if __name__ == "__main__":
    train, test = load_data()
    # Imprimir las dimensiones de los datasets para confirmar que están cargando correctamente
    print("Train Data Shape:", train.shape)
    print("Test Data Shape:", test.shape)




