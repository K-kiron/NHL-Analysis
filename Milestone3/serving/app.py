import os
import logging
from flask import Flask, jsonify, request
import pandas as pd
import joblib
from comet_ml import API

# To specify time at which logs were sent
# Helps for tracking purposes
import datetime
import time

parent_model_path = "./"

app = Flask(__name__)

LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

api_key = os.environ.get("COMET_API_KEY")
api = API(api_key=api_key)

global model, model_name

# Loading default model ("log_reg_basemodel_distance")
# if not os.path.exists(parent_model_path+"log_reg_basemodel_distance.pkl"):
initial_files = set(os.listdir(parent_model_path))

experiment = api.get_model(workspace="ift6758b-project-b10", model_name="log_reg_basemodel_distance")
experiment.download("1.1.0", parent_model_path, expand=True)

model_name = "log_reg_basemodel_distance"
updated_files = set(os.listdir(parent_model_path))

pkl_file = (updated_files - initial_files).pop()
os.rename(pkl_file, model_name + ".pkl")

model = joblib.load(model_name+".pkl")

@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

# RUN : curl http://IP_ADDRESS:PORT/logs
@app.route("/logs", methods=["GET"])
def logs():
    try:
        # TODO: read the log file specified and return the data
        response = {}
        with open(LOG_FILE, "r") as file:
            for i in file:
                response[i] = i
        #    LOGS = file.read()
        #return jsonify({f'/logs endpoint @ {datetime.datetime.now()}': LOGS})
        return jsonify(response)
    except Exception as e:
        return jsonify({f'You have encountered the following ERROR @ {datetime.datetime.now()}': str(e)})

#Ex: curl -X POST -H "Content-Type: application/json" -d '{"workspace": "ift6758b project b10", "project": "nhl-project-b10", "model": "adaboost-max-depth-1-v2", "version": "1.0.1"}' http://IP_ADDRESS:PORT/download_registry_model
@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    
    try:
        global model, model_name
        status_code = 400
        json_data = request.get_json()
        app.logger.info(f"Model in use before download: {model_name}")

        workspace = json_data.get('workspace')  # Is "ift6758b-project-b10"
        model_candidate = json_data.get('model')     # REGISTERED model name (ex: "adaboost-max-depth-1-v2")
        version = json_data.get('version')      # model version (ex: "1.0.1")
        api_key = os.environ.get("COMET_API_KEY")
        if not api_key:
            response = {'NOTIFICATION': "Unable to retrieve ${COMET_API_KEY} environment variable"}
            app.logger.info(response)
        api = API(api_key=api_key)


        # TODO: check to see if the model you are querying for is already downloaded
        model_path = f"{model_candidate}.pkl"
        if os.path.exists(model_path):
            # TODO: if yes, load that model and write to the log about the model change.
            # eg: app.logger.info(<LOG STRING>)
            model = joblib.load(model_path)
            response = {'NOTIFICATION': f"Currently loaded model {model_candidate} from LOCAL @ {datetime.datetime.now()}"}
            model_name = model_candidate
            status_code = 200
            app.logger.info(response)
        # TODO: if no, try downloading the model
        else:
            initial_files = set(os.listdir(parent_model_path))
            if workspace and model_candidate and version:
                try:
                    # if it succeeds, load that model and write to the log
                    # see Comet API documentation
                    # https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/APIExperiment/#apiexperimentdownload_model
                    experiment = api.get_model(workspace=workspace, model_name=model_candidate)
                    experiment.download(version, parent_model_path, expand=True)

                    # Making sure the .pkl file that is downloaded has appropriate naming for further retrieval
                    time.sleep(2)
                    updated_files = set(os.listdir(parent_model_path))

                    pkl_file = (updated_files - initial_files).pop()
                    os.rename(pkl_file, model_candidate+".pkl")

                    model = joblib.load(model_path)
                    response = {'NOTIFICATION': f"Currently loaded model {model_candidate} from CometML DOWNLOAD @ {datetime.datetime.now()}"}
                    status_code = 200
                    model_name = model_candidate
                    app.logger.info(response)
                except Exception as e:
                    # If it fails, write to the log about the failure and keep the currently loaded model
                    response = {'NOTIFICATION': f"ERROR (Failure to download) @ {datetime.datetime.now()}: {e}"}
                    app.logger.error(response)
            else:
                response = {'NOTIFICATION': f"ERROR (Invalid model details) @ {datetime.datetime.now()}"}
                app.logger.error(response)

        #response = {'NOTIFICATION': f'Model {model_name} loaded successfully either LOCALLY or from CometML DOWNLOAD @ {datetime.datetime.now()}'}
        app.logger.info(f"Model in use: {model_name}")
        return jsonify(response), status_code

    except Exception as e:
        return jsonify({f'You have encountered the following ERROR @ {datetime.datetime.now()}': str(e)}), 500

# RUN: curl -X POST -H "Content-Type: application/json" --data @input.json http://IP_ADDRESS:PORT/predict
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    try:
        # Need to run /download_registry_model prior
        global model, model_name
        model = joblib.load(parent_model_path+model_name+".pkl")
        json_data = request.get_json()
        app.logger.info(json_data)
        
        df = pd.DataFrame.from_dict(json_data, orient='columns')

        if model_name == 'log_reg_basemodel_distance':
            df = df[['shotDistance']]
        elif model_name == 'log_reg_basemodel_angle':
            df = df[['shotAngle']]
        elif model_name == 'log_reg_basemodel_distance_angle':
            df = df[['shotDistance','shotAngle']]

        predictions = model.predict_proba(df)[:,1]
        response = {'MODEL predictions': predictions.tolist(), 'features_used': df.columns.values.tolist()}
        app.logger.info(response)
        return jsonify(response), 200

    except Exception as e:
        return jsonify({f'You have encountered the following ERROR @ {datetime.datetime.now()}': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)