import json
import requests
import pandas as pd
import logging


logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """

        response = requests.post(
            f"{self.base_url}/predict", data=json.dumps(X.to_dict(orient="records"))
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server responded with error: {response.text}")
        
        return pd.DataFrame(response.json())

    def logs(self) -> dict:
        """Get server logs"""

        response = requests.get(f"{self.base_url}/logs")
        if response.status_code != 200:
            raise RuntimeError(f"Server responded with error: {response.text}")

        return response.json()

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        response = requests.post(
            f"{self.base_url}/download_registry_model",
            data=json.dumps({"workspace": workspace, "model": model, "version": version}),
        )

        if response.status_code != 200:
            raise RuntimeError(f"Server responded with error: {response.text}")

        return response.json()



client = ServingClient(ip='127.0.0.1', port=5000)

# Test predict method
test_data = pd.DataFrame({'shotDistance': [78], 'shotAngle': [2]})
prediction = client.predict(test_data)
print(prediction)

# Test logs method
logs = client.logs()
print(logs)

# Test download_registry_model method
model_info = client.download_registry_model('workspace', 'model', 'version')
print(model_info)