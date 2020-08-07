<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/andronikmk/toxic-content-monitoring/master/docs/_static/Big-Armor-Logo.png" width="142"/>
    <br>
<p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/andronikmk/toxic-content-monitoring/master)



### Table of Contents
- [Introduction to Toxic Content Monitoring](#introduction-to-toxic-content-monitoring)
- [Install](#install)
  - [From source](#from-source)
  - [Create enviornment with conda and pip](#create-enviornment-with-conda-and-pip)
- [Run it](#run-it)
- [Check it](#check-it)
- [Interactive API docs](#interactive-api-docs)
- [Docker](#docker)


# Introduction to Toxic Content Monitoring

While there exist systems for detecting toxicity, suicidality, and other concerning behaviors, 
they are all either whole-system programs or are limited to only a small number of topics. 
This project creates an API that can be integrated into an applications used by anyone, such as 
a social worker, to detect concerning behaviors. It gives the end-user the ability to send 
information to the API, and get meaningful results.

# Install
This repo is tested on Python 3.7+, PyTorch 1.0.0+ (PyTorch 1.3.1+ for examples) and TensorFlow 2.0.

You should install Toxic Content Monitoring in a virtual [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
This project was created using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html), if you are infamilier with [Anaconda](https://www.anaconda.com/), please reference this [cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) after you install you choice of Anaconda distrubution.

## From source
```bash
git clone https://github.com/andronikmk/toxic-content-monitoring.git
cd toxic-content-monitoring
```
## Create enviornment with conda and pip
Make sure to be in the correct directory.
```bash
# create conda env.
conda create -n toxic-content-monitoring python==3.7

# activate env
conda activate toxic-content-monitoring

# pip install dependencies
pip install -r requirements.txt

# add JSON object to an iPython file.
python -m ipykernel install --user --name toxic-content-monitoring --display "toxic-content-monitoring (Python3)"

```

# Run it
```console
$ uvicorn src.main:app --host 0.0.0.0

INFO:     Started server process [67600]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

# Check it
The [FastAPI](https://fastapi.tiangolo.com/) interface will be located on the docs page, i.e. http://0.0.0.0:8000/docs

You will see the JSON response as:
```JSON
{
  "text": "string"
}
```

# Interactive API docs

Now go to http://0.0.0.0:8000/docs

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/andronikmk/toxic-content-monitoring/master/docs/_static/fastapi.png"/>
    <br>
<p>

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/andronikmk/toxic-content-monitoring/master/docs/_static/screenshot2.png"/>
    <br>
<p>

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/andronikmk/toxic-content-monitoring/master/docs/_static/screenshot3.png"/>
    <br>
<p>

# Docker
In addition, you can pull the [docker image](https://hub.docker.com/r/andronikmk/toxic-content-monitoring) to you local machine and run or locally. Or pull it to a
instance on Amazon AWS, GCP, etc..

```bash
# Pull the docker image to local machine.
docker pull andronikmk/toxic-content-monitoring:0.1

# Run the docker image
docker run -it andronikmk/toxic-content-monitoring:0.1
```
```console
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```