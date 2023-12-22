# IFT6758B-Project

### Conda 

Conda uses the provided `environment.yml` file.
You can ignore `requirements.txt` if you choose this method.
Make sure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual) installed on your system.
Once installed, open up your terminal (or Anaconda prompt if you're on Windows).
Install the environment from the specified environment file:

    conda env create --file environment.yml
    conda activate B10-ift6758-conda-env

After you install, register the environment so jupyter can see it:

    python -m ipykernel install --user --name=B10-ift6758-conda-env

You should now be able to launch jupyter and see your conda environment:

    jupyter-lab

If you make updates to your conda `environment.yml`, you can use the update command to update your existing environment rather than creating a new one:

    conda env update --file environment.yml    

You can create a new environment file using the `create` command:

    conda env export > environment.yml

### Docker
The repo root contains the `docker-compose.yaml`, outlining the service architecture for the application.<br><br>
The following steps must be followed to run the application:
1. Set your experiment key as an environment variable
   1. On Linux/MacOS: `export COMET_API_KEY=<API_KEY>`. Additionally, execute `source <bashfile>` to ensure that the environment variables are sourced correctly.
   2. On Windows: `set COMET_API_KEY=<API_KEY>`<br>
2. Start the Docker daemon on your machine.
3. Ensure that the terminal is in the repository's root directory (`IFT6758B-Project-B10`).
4. Exeecute the command `docker-compose up` to launch up the services.
5. Once the containers are spawned, access the UI at `http://0.0.0.0:8050`.