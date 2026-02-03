
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Union


def load_model(model_path: Union[str, Path]):
    """Load a trained model from a pickle file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict_single(model, passenger_data: Dict) -> int:
    df = pd.DataFrame([passenger_data])
    prediction = model.predict(df)
    return int(prediction[0])





def predict_batch(model, passengers_df: pd.DataFrame) -> pd.DataFrame:
    
    predictions = model.predict(passengers_df)
    
    result = passengers_df.copy()
    result['Survived'] = predictions    
    return result


# Example usage
if __name__ == "__main__":
    # Load the trained model
    model = load_model("results/titanic_best_model.pkl")
    
    # Example 1: Single passenger prediction
    # This represents a passenger from the ACTUAL engineered dataset
    passenger_example = {
        'Unnamed: 0': 0,
        'Pclass': 3,
        'Sex': 1,  # 1 = male, 0 = female
        'HasCabin': 0,
        'FamilySize': 1,
        'IsAlone': 1,
        'Title_Master': 0,
        'Title_Miss': 0,
        'Title_Mr': 1,
        'Title_Mrs': 0,
        'Title_Rare': 0,
        'AgeGroup_Child': 0,
        'AgeGroup_Teen': 0,
        'AgeGroup_Adult': 1,
        'AgeGroup_Senior': 0,
        'FareGroup_Low': 1,
        'FareGroup_Medium': 0,
        'FareGroup_High': 0,
        'Embarked_C': 0,
        'Embarked_Q': 0,
        'Embarked_S': 1,
    }
    
    prediction = predict_single(model, passenger_example)
    
    print(f"Passenger Prediction: {'Survived' if prediction == 1 else 'Did not survive'}")
    
    # Example 2: Create different passenger profiles
    print("\n" + "="*50)
    print("Testing different passenger profiles:")
    print("="*50)
    
    # Profile 1: First class woman with cabin
    first_class_woman = {
        'Unnamed: 0': 1,
        'Pclass': 1,
        'Sex': 0,  # Female
        'HasCabin': 1,
        'FamilySize': 2,
        'IsAlone': 0,
        'Title_Master': 0,
        'Title_Miss': 0,
        'Title_Mr': 0,
        'Title_Mrs': 1,
        'Title_Rare': 0,
        'AgeGroup_Child': 0,
        'AgeGroup_Teen': 0,
        'AgeGroup_Adult': 1,
        'AgeGroup_Senior': 0,
        'FareGroup_Low': 0,
        'FareGroup_Medium': 0,
        'FareGroup_High': 1,
        'Embarked_C': 1,
        'Embarked_Q': 0,
        'Embarked_S': 0,
    }
    
    pred1 = predict_single(model, first_class_woman)
    print(f"\n1st Class Woman: {'Survived' if pred1 == 1 else 'Did not survive'} ")

    # Profile 2: Third class man, no cabin
    third_class_man = {
        'Unnamed: 0': 2,
        'Pclass': 3,
        'Sex': 1,  # Male
        'HasCabin': 0,
        'FamilySize': 1,
        'IsAlone': 1,
        'Title_Master': 0,
        'Title_Miss': 0,
        'Title_Mr': 1,
        'Title_Mrs': 0,
        'Title_Rare': 0,
        'AgeGroup_Child': 0,
        'AgeGroup_Teen': 0,
        'AgeGroup_Adult': 1,
        'AgeGroup_Senior': 0,
        'FareGroup_Low': 1,
        'FareGroup_Medium': 0,
        'FareGroup_High': 0,
        'Embarked_C': 0,
        'Embarked_Q': 0,
        'Embarked_S': 1,
    }
    
    pred2 = predict_single(model, third_class_man)
    print(f"3rd Class Man: {'Survived' if pred2 == 1 else 'Did not survive'}")
    
    # Profile 3: Child (Master title) with family
    child = {
        'Unnamed: 0': 3,
        'Pclass': 2,
        'Sex': 1,  # Male
        'HasCabin': 0,
        'FamilySize': 3,
        'IsAlone': 0,
        'Title_Master': 1,  # Child
        'Title_Miss': 0,
        'Title_Mr': 0,
        'Title_Mrs': 0,
        'Title_Rare': 0,
        'AgeGroup_Child': 1,
        'AgeGroup_Teen': 0,
        'AgeGroup_Adult': 0,
        'AgeGroup_Senior': 0,
        'FareGroup_Low': 0,
        'FareGroup_Medium': 1,
        'FareGroup_High': 0,
        'Embarked_C': 0,
        'Embarked_Q': 0,
        'Embarked_S': 1,
    }
    
    pred3 = predict_single(model, child)
    print(f"Child (Master): {'Survived' if pred3 == 1 else 'Did not survive'} ")
