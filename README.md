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
The source files are located in `src` directory:


Details will be added.




## Citation

Please cite the paper, If you used the codes in this repository.
