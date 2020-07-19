# toxic-content-monitoring-research
This directory contains an attempt to do neural architecture search through NNI. I was not able to finish it in time. It seems pretty close to working, but there's some type error I wasn't able to fix in `custom_darts_trainer.py`.

Note:

To use this, one cannot simply install pytorch and NNI using pipenv. Instead, the following must be done;

* install `nni`, `dill`, and `pandas`, through pipenv, if you want.
* install pytorch and torchvision through `pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html`.
* install torchnlp through `pip install pytorch-nlp`.
