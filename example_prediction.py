
import sys
sys.path.append('src')

from predict import load_model, predict_single
import pandas as pd


def main():
    print("TITANIC SURVIVAL PREDICTION - DEMO")
    

    # Load the trained model
    print("\n1. Loading trained model...")
    try:
        model = load_model("results/titanic_best_model.pkl")
        print("✓ Model loaded successfully!")
    except FileNotFoundError:
        print("✗ Model not found. Please run src/pipeline.py first to train the model.")
        return
    
    print("\n2. Preparing passenger data...")
    
    # Example passengers (using ACTUAL engineered features from Phase 3)
    # Features: Unnamed: 0, Pclass, Sex, HasCabin, FamilySize, IsAlone, 
    #           Title_Master, Title_Miss, Title_Mr, Title_Mrs, Title_Rare,
    #           AgeGroup_Child, AgeGroup_Teen, AgeGroup_Adult, AgeGroup_Senior,
    #           FareGroup_Low, FareGroup_Medium, FareGroup_High,
    #           Embarked_C, Embarked_Q, Embarked_S
    
    passengers = [
        {
            'name': 'Rose (1st Class Woman with Cabin)',
            'data': {
                'Unnamed: 0': 0,       # Index
                'Pclass': 1,           # First class
                'Sex': 0,              # Female
                'HasCabin': 1,         # Has cabin
                'FamilySize': 2,       # Has family
                'IsAlone': 0,          # Not alone
                'Title_Master': 0,
                'Title_Miss': 0,
                'Title_Mr': 0,
                'Title_Mrs': 1,        # Mrs. title
                'Title_Rare': 0,
                'AgeGroup_Child': 0,
                'AgeGroup_Teen': 0,
                'AgeGroup_Adult': 1,   # Adult
                'AgeGroup_Senior': 0,
                'FareGroup_Low': 0,
                'FareGroup_Medium': 0,
                'FareGroup_High': 1,   # High fare
                'Embarked_C': 1,       # Cherbourg
                'Embarked_Q': 0,
                'Embarked_S': 0,
            }
        },
        {
            'name': 'Jack (3rd Class Man, No Cabin)',
            'data': {
                'Unnamed: 0': 1,
                'Pclass': 3,           # Third class
                'Sex': 1,              # Male
                'HasCabin': 0,         # No cabin
                'FamilySize': 1,       # Solo
                'IsAlone': 1,          # Alone
                'Title_Master': 0,
                'Title_Miss': 0,
                'Title_Mr': 1,         # Mr. title
                'Title_Mrs': 0,
                'Title_Rare': 0,
                'AgeGroup_Child': 0,
                'AgeGroup_Teen': 0,
                'AgeGroup_Adult': 1,   # Adult
                'AgeGroup_Senior': 0,
                'FareGroup_Low': 1,    # Low fare
                'FareGroup_Medium': 0,
                'FareGroup_High': 0,
                'Embarked_C': 0,
                'Embarked_Q': 0,
                'Embarked_S': 1,       # Southampton
            }
        },
        {
            'name': 'Child (Master) with Family',
            'data': {
                'Unnamed: 0': 2,
                'Pclass': 2,           # Second class
                'Sex': 1,              # Male
                'HasCabin': 0,         # No cabin
                'FamilySize': 4,       # Large family
                'IsAlone': 0,          # With family
                'Title_Master': 1,     # Master (young boy)
                'Title_Miss': 0,
                'Title_Mr': 0,
                'Title_Mrs': 0,
                'Title_Rare': 0,
                'AgeGroup_Child': 1,   # Child
                'AgeGroup_Teen': 0,
                'AgeGroup_Adult': 0,
                'AgeGroup_Senior': 0,
                'FareGroup_Low': 0,
                'FareGroup_Medium': 1, # Medium fare
                'FareGroup_High': 0,
                'Embarked_C': 0,
                'Embarked_Q': 0,
                'Embarked_S': 1,
            }
        },
        {
            'name': 'Young Woman (3rd Class, Miss)',
            'data': {
                'Unnamed: 0': 3,
                'Pclass': 3,           # Third class
                'Sex': 0,              # Female
                'HasCabin': 0,         # No cabin
                'FamilySize': 1,       # Solo
                'IsAlone': 1,          # Alone
                'Title_Master': 0,
                'Title_Miss': 1,       # Miss title
                'Title_Mr': 0,
                'Title_Mrs': 0,
                'Title_Rare': 0,
                'AgeGroup_Child': 0,
                'AgeGroup_Teen': 1,    # Teenager
                'AgeGroup_Adult': 0,
                'AgeGroup_Senior': 0,
                'FareGroup_Low': 1,    # Low fare
                'FareGroup_Medium': 0,
                'FareGroup_High': 0,
                'Embarked_C': 0,
                'Embarked_Q': 1,       # Queenstown
                'Embarked_S': 0,
            }
        }
    ]
    
    print("\n3. Making predictions...")

    
    results = []
    for passenger in passengers:
        prediction = predict_single(model, passenger['data'])
        
        result = {
            'Name': passenger['name'],
            'Prediction': 'SURVIVED' if prediction == 1 else 'DID NOT SURVIVE',
        }
        results.append(result)
    
    # Display results as a table
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    


if __name__ == "__main__":
    main()
