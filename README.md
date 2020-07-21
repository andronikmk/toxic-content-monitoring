<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/andronikmk/toxic-content-monitoring/master/docs/_static/Big-Armor-Logo.png" width="142"/>
    <br>
<p>

<p align="center">
    <a href="https://github.com/andronikmk/toxic-content-monitoring/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/badge/License-MIT-yellow.svg">
    </a>
</p>

# Toxic Content Monitoring

While there exist systems for detecting toxicity, suicidality, and other concerning behaviors, 
they are all either whole-system programs or are limited to only a small number of topics. 
This project creates an API that can be integrated into an applications used by anyone, such as 
a social worker, to detect concerning behaviors. It gives the end-user the ability to send 
information to the API, and get meaningful results.

# Installation
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
python -m ipykernel install --user --name toxic-content-monitoring  --display "toxic-content-monitoring (Python3)"

```

## Run it
```console
$ uvicorn src.main:app --host 0.0.0.0

INFO:     Started server process [67600]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```