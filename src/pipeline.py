import pickle 
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from config import PipelineConfig
from data import load_csv
from features import prepare_features
from train import split_data , train_model, get_models
from evaluate import evaluate_model , pick_best
import train

# main function to run the pipeline
def run_pipeline(config: PipelineConfig )->None:
    #loading our data
    df = load_csv(config.train_path)
    #splitting feature and labels
    X , y = prepare_features(df,config.target_col) 
    # train / test split
    x_train , x_test , y_train , y_test = split_data (X , y , config.test_size , config.random_state)
    #creating a scaler to scale data for KNN and SVM
    scaler = MinMaxScaler(feature_range=(0,1))
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    models = get_models(config.random_state)    # models dictionnary 
    results = {}   # results dict to store models accuracy
    for name , model in models.items():
        if name in {'KNN','SVM'}:
            # use scaled data
            trained = train_model(model,x_train_scaled,y_train)
            results[name] = evaluate_model(trained , x_test , y_test)
        else:
            # Other models use unscaled data
            trained = train_model(model,x_train,y_train)
            results[name] = evaluate_model(trained , x_test , y_test)
    best_name , best_result = pick_best(results)
    print("\nBest model:", best_name)
    print("Accuracy:", f"{best_result['accuracy']:.4f}")
    
    # save the best model
    if config.save_model : 
        config.model_output_path.parent.mkdir(parents=True,exist_ok=True)
        best_model = models [best_name]
        with open (config.model_output_path,"wb") as f : 
            pickle.dump(best_model,f)
        print("Saved best model to:", config.model_output_path)

# MAIN FUNCTION
if __name__=="__main__" :
    cfg = PipelineConfig ()
    run_pipeline(cfg)