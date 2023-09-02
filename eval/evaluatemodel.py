import argparse
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import shutil
from azureml.core import Run
from azureml.core import Model

    
    
# define functions
def main(args):
    
    model_name = args.model_name
    
    run = Run.get_context()
    ws = run.experiment.workspace
    run_id = run.id
    print('run_id =' + run_id)
    model_list = Model.list(ws, name=model_name, latest=True)
    first_registration = len(model_list)==0
    current_model = None
        
    # read in data
    print('about to read file:' + args.test_data)
    X_test = pd.read_csv(args.test_data + '/X_test.csv')
    df_y_test = pd.read_csv(args.test_data + '/Y_test.csv')
    y_test  = df_y_test.values.flatten()
    #load champion model
    model_file = os.path.join(args.model_folder, 'titanic_model.pkl')
    champion_model = joblib.load(model_file)
    
    y_pred_current = champion_model.predict(X_test)
    print('y_pred_current')
    print(y_pred_current)
    print('y_test')
    print(y_test)
    
    champion_auc = roc_auc_score(y_test,y_pred_current)
    print('champion_auc:' , champion_auc)
    
    champion_acc = np.average(y_pred_current == y_test)
    print('champion_acc:', champion_acc)

    
    try:
        current_model_aml = Model(ws,args.model_name)
        os.makedirs("current_model", exist_ok=True)
        current_model_aml.download("current_model",exist_ok=True)
        current_model = mlflow.sklearn.load_model(os.path.join("current_model",args.model_name))
    except:
        print('no model register with name' + args.model_name)
        pass
    
    if current_model:
        y_pred_current = current_model.predict(X_test)
        current_acc = np.average(y_pred_current == y_test)
        print('current_acc:', current_acc)
        if champion_acc >= current_acc:
            print('better model found, registering')

            # Signature
            signature = infer_signature(X_test, y_test)

            # Conda environment
            custom_env =_mlflow_conda_env(
                additional_conda_deps=["scikit-learn==1.1.3"],
                additional_pip_deps=["mlflow<=1.30.0"],
                additional_conda_channels=None,
            )

            # Sample
            input_example = X_test.sample(n=1)

            # Log the model manually
            mlflow.sklearn.log_model(champion_model, 
                                     artifact_path=args.model_name, 
                                     conda_env=custom_env,
                                     signature=signature,
                                     input_example=input_example)
            model_uri = f'runs:/{run_id}/{args.model_name}'
            mlflow.register_model(model_uri,args.model_name)
            ##########################################################
    
            
        else:
            print('current model performs better than champion model ')
            print('champion_acc:', champion_acc)
            print('current_acc:', current_acc)
    else:
        print('no current model')
        print("First time model train, registering")

        signature = infer_signature(X_test, y_test)

        # Conda environment
        custom_env =_mlflow_conda_env(
                additional_conda_deps=["scikit-learn==1.1.3"],
                additional_pip_deps=["mlflow<=1.30.0"],
                additional_conda_channels=None,
            )

        # Sample
        input_example = X_test.sample(n=1)

        # Log the model manually
        mlflow.sklearn.log_model(champion_model, 
                                 artifact_path=args.model_name, 
                                 conda_env=custom_env,
                                 signature=signature,
                                 input_example=input_example)
        model_uri = f'runs:/{run_id}/{args.model_name}'
        mlflow.register_model(model_uri,args.model_name)
        

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--test_data", default="data", type=str, help="Path to test data")
    parser.add_argument("--model_folder", default="data", type=str, help="Path to model data")
    parser.add_argument("--model_name",default='mmchapter9titanic',type=str, help="Name of the model in workspace")

    # parse args
    args = parser.parse_args()
    
    print(args.test_data)
    print(args.model_folder)
    print(args.model_name)
        
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
