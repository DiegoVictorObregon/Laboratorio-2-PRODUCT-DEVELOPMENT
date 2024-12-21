import pandas as pd

def handle_missing_values(data):
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    return data

def encode_categorical_variables(data):
    data = pd.get_dummies(data, columns=['Sex', 'Embarked'], drop_first=True)
    return data

def create_new_features(data):
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    return data

def drop_unwanted_columns(data):
    # Asegúrate de eliminar cualquier columna que no quieras incluir en el modelo
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    return data

def prepare_features():
    train_path = 'data/raw/train.csv'
    test_path = 'data/raw/test.csv'
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    train_data = handle_missing_values(train_data)
    train_data = encode_categorical_variables(train_data)
    train_data = create_new_features(train_data)
    train_data = drop_unwanted_columns(train_data)
    
    test_data = handle_missing_values(test_data)
    test_data = encode_categorical_variables(test_data)
    test_data = create_new_features(test_data)
    test_data = drop_unwanted_columns(test_data)
    
    # Guardar los datos procesados
    train_data.to_pickle('data/processed/train_featured.pkl')
    test_data.to_pickle('data/processed/test_featured.pkl')

if __name__ == "__main__":
    prepare_features()





