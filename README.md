# iDEM_writing_assistant
The iDEM writing assistant T3.4

## GPU support

It is highly recommended to run the writing assistant on a GPU. For doing this, check if the NVIDIA CUDA drivers are installed
```commandline
nvidia-smi
```
If it is not installed, this is the way to install it on an Amazon EC2 instance with Ubuntu Linux installed, other Linux
systems will have similar ways to install the drivers:
```
sudo apt-get update
sudo apt-get upgrade
sudo apt install nvidia-cuda-toolkit
sudo apt install nvidia-driver-570
```
Check if cuda is available for python:

Open a python shell
```
python3
```

Then check if cuda is available for torch:
```commandline
import torch
torch.cuda.is_available()
```

## Python version

This software has been tested with python 3.12. Older python versions, especially below python 3.10 may not be 
valid for the purpuse of running the writing assistant.

## Install requirement and spacy models
```
pip3 install -r requirements
```
You may have to install a special version of *pytorch* that has support for *gemma3*
```
pip3 install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```
Install the necessary spacy language model for low level language processing
python -m spacy download es_core_news_sm 

## Praparing the data dir

Data downloads:
Data must stored in ~/data/iDEM/

For Spanish download or checkout the COWSL2H data from  
https://github.com/ucdaviscl/cowsl2h
into ~/data/iDEM/writing_assistant/. 
THis should create a path ~/data/iDEM/writing_assistant/cowsl2h-master/ which contains the data.

## Basic usage

Usage: 
```python3 scripts/create_corrections.py --help ```

An access token for Hugging Face will have to be provided with --access_token

## Outputs

TSV files with the outputs are written to files the folder `./outputs`. Filenames contain information on the 
LLM used, the language and the device. For example `LLAMA1B_es_cuda.tsv` uses the 1B version of LLAMA on Spanish and by
using the GPU via cuda

## Evaluation
Evaluation can be run on the outpufiles created by `scripts/create_corrections.py`. Splitting generation of output and
evaluation saves resources since it is often unnecessary to re-run the creation of outputs. 

```
python3 scripts/evaluate_cowsl_outputs.py [outputfiles]
```

Bleurt can be computed with 
```
python3 scripts/apply_bleurt2output.py [outputfiles]
```
