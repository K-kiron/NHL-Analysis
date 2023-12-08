<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- CONTRIBUTING -->
## RUNNING FLASK APP

To use the Flask app (`../Milestone3/serving/app.py`), you must:

1. Open terminal and cd to `../Milestone3/serving/`. Before running the Flask app, run `export COMET_API_KEY="example_key_1234"` to specify your experiment key as an environment variable. Then run `python app.py`
2. To access endpoints `/logs`, `/download_registry_model`, or `/predict`, open a new terminal and follow the subsequent instructions
3. To get `/logs`, run `curl http://IP_ADDRESS:PORT/logs`
4. To load a model (either from Comet or locally from `.pkl` file stored in `../../serving/`), run `curl -X POST -H "Content-Type: application/json" -d '{"workspace": "ift6758b-project-b10", "model": "MODEL_NAME", "version": "NUMBER.NUMBER.NUMBER"}' http://IP_ADDRESS:PORT/download_registry_model`. Note that all the comet retrieval info can be stored in a `.json` file (stored in `/serving`). This json file would have the following structure, for example:
```
{
    "workspace": "ift6758b-project-b10",
    "model": "adaboost-max-depth-1-v2",
    "version": "1.0.1"
}
```
Take care to save the file with UTF-8 encoding. You can thereby load a model by running the command `curl -X POST -H "Content-Type: application/json" -d @PATH/TO/.../serving/FILENAME.json http://IP_ADDRESS:PORT/download_registry_model`. 
5. To obtain a model prediction from a model downloaded from COMET or loaded locally using Flask app inside of a Jupyter notebook, add the following cell:
```
X = pd.DataFrame_input_df()
r = requests.post(
    "http://IP_ADDRESS:PORT/predict", 
    json=json.loads(X.to_json())
)
```
6. To obtain a model prediction from a model downloaded from COMET or loaded locally using Flask app directly from the terminal, run `curl -X POST -H "Content-Type: application/json" --data @JSON_FILENAME.json http://IP_ADDRESS:PORT/predict`. Note that JSON_FILENAME.json must be specified. 
