# DTT: Tabular Transformer for Joinability by Leveraging Large Language Models

This repository contains resources developed within the following paper:

    A. Dargahi Nobari, and D. Rafiei. “DTT: An Example-Driven Tabular Transformer by Leveraging Large Language Models”.
	
You may check the [paper](https://arxiv.org/abs/2303.06748) ([PDF](https://arxiv.org/pdf/2303.06748)) for more information.


## Requirements

Several libraries are used in the project. You can use the provided `environment.yml` file to create the conda environment for the project. 
If you prefer not to use the environment file, the environment can be set up by the following command.
```
conda create -n dtt python=3.10
conda activate dtt
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 torchtext==0.12.0 cudatoolkit=11.3 -c pytorch
conda install -c huggingface transformers==4.21.1
conda install -c conda-forge pytorch-lightning==1.7.3 
conda install pandas==1.4.1  nltk==3.7
conda install pip
pip install SentencePiece==0.1.97
pip install openai==0.26.5 # to test with GPT3
```


## Usage

Three main directories are in the repo: `models`, `data`, and `src`.

### models
This directory contains the trained model. Due to the size limit of github, this folder is not added to the repository. Our trained model with default settings can be downloaded [here](https://drive.google.com/file/d/1_7xtf9p7DZqPxbjYkRsI_Sylk2wYm34G/view?usp=share_link).

### data
All datasets are included in this directory. Please extract `data.tar.gz` to access the content of this folder. The datasets are in `Datasets` directory. Each dataset contains several tables (each as a folder) and each table contains `source.csv`, `target.csv`, and `ground truth.csv`. The datasets available in this file are:
- `FF_AJ`: The web tables and spreadsheet dataset. Tables starting with `AJ` prefix belong to the web tables and those starting with `FF` belong to the spreadsheet dataset.
- `Synthetic_basic_10tr_100rows__08_35len`: The _Syn_ dataset, reported in the table. It contains 10 tables of synthetic tabular transformations.
- `Single_Replace_05tr_050rows__08_35len`: The _Syn-RP_ dataset, reported in the table. It contains 5 tables of simple synthetic tabular transformations.
- `Single_Substr_05tr_050rows__08_35len`: The _Syn-ST_ dataset, reported in the table. It contains 5 tables of medium (substring) synthetic tabular transformations.
- `Single_Replace_05tr_050rows__08_35len`: The _Syn-RV_ dataset, reported in the table. It contains 5 tables of difficult (reverse) synthetic tabular transformations.
- `_replace_lens`, `_reverse_lens`, `_substr_lens` are **groups** of datasets similar to the _Syn-RP_, _Syn-ST_, and _Syn-RV_ datasets but with input length varying from 5 to 60. 

Before using each dataset to test the model, it should be broken to training examples and test set.


### src
The source files are located in `src` directory. This directory contains three sub-directories.


##### data_processor
Files in this folder are to generate, pre-process and use datasets.
* `table2sample.py`: It takes a dataset as input and transforms it into the sample set that can be used for training the model.
* `table_breaker`: See details for each file.
  * `table_breaker.py`: This file will break a dataset into training examples and a test set.
  * `run_breaker.sh`: This file contains examples of using `table_breaker.py` and its command line arguments, as well as using it for many datasets automatically.
* `synthetic_generator/string_transformations`: To generate synthetic datasets. See details for each file.
  * `Transformation`: Just libraries of transformations. Not to be used directly. 
  * `basic_generator.py`: This file is used to generate a synthetic sample set that is used to train the model. 
  * `single_basic_generator.py`: This file generates synthetic datasets with one transformation (such as _Syn-RP_, _Syn-ST_, and _Syn-RV_ datasets) that are used to test the model.
  * `synthetic_basic_generator.py`: This file generates synthetic datasets with several transformations (such as _Syn_ dataset) that is used to test the model.
* `synthetic_generator/noise_generator`: To manually add noise to the datasets. See details for each file.
  * `dataset_noiser.py`: To corrupt a specific percentage of rows, adding noise to datasets and saving them as a new dataset.
  * `run_noiser.sh`: This file contains examples of using `dataset_noiser.py` and its command line arguments, as well as using it for many datasets.

##### deep_models
This directory includes files to train and use the model.

* `byt5`: To Train and run a basic test with the main model (based on ByT5 model). See details for each file.
  * `byt5trainer.py`: This file will train (finetune) a model given a set of training data.
  * `load_model.py`: A basic example of using the trained model for transformation. If you are interested in running the model without any finetuning or changes, just download the pretrained model provided in the models section, set the model path in `MODEL_PATH` variable and run the file.
* `JoinEval.py`: Libraries to evaluate the performance of table joining. 
* `Util.py`: Some common functions.
* `tester.py`: This is a large file used to test the model on various datasets. The parameters can be set inside the code or be passed via command line arguments. There are three files that exemplify how the tester file may be used:
  * `run_tester.sh`: This file is an example of how a single instance of `tester.py` can be called. Also, it contains shell code to test several models on various datasets. 
  * `run_len_tester.sh`: This file is an example of running `tester.py` on datasets with various input lengths.
  * `run_noise_tester.sh`: This file is an example of running `tester.py` on datasets that had been manually noised. The only difference is in the dataset directory.
  * `run_gpt_tester.sh`: This file uses `tester.py` with GPT-3 model. Please make sure your OpenAI API key is stored in `openai.key` file in `deep_models` directory.
  * `run_two_model_tester.sh`: This file uses `tester.py` with a combined setting for DTT and GPT-3 model. Please make sure your OpenAI API key is stored in `openai.key` file in `deep_models` directory.


##### analyzer
The codes in this directory just summarize the results of the model into short tables.
- `result_tlb_summary.py`: get the summary of output when several models are used to transform the data.
- `len_tlb_summary.py`: get the summary of output when several input lengths are experimented.


## Citation

Please cite the paper, If you used the codes in this repository.
