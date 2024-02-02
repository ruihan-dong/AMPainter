# AMPainter
Painting the peptides with antimicrobial attibutes via deep reinforcement & hypergraph learning

(V1 upload: 24-02-01 by Ruihan Dong)

## Overview
This framework consists of three modules:
* a policy network to assign the mutation sites
* a fine-tuned protein language model to replace the assigned residues
* a predictor named HyperAMP to evaluate the antimicrobial activity


## Requirements
* pytorch
* [DHG](https://github.com/iMoonLab/DeepHypergraph)
* [ankh](https://github.com/agemagician/Ankh)
* transformers
* sklearn
* Levenshtein
* modlamp


## Usage
### 1) HyperAMP solely
#### Train predictor
```bash
cd HyperAMP
python ankh_embedding.py
python tfidf.py
python train.py # or train_5folds.py
```
#### Predict
```bash
python ./HyperAMP/predict.py  # change input file path first
```
Length of a input peptide sequence should be less than 40.

The output of a input sequence is the transformed score based on its logMIC value to represent its antimicrobial activity.

### 2) Fine-tune Ankh
#### Data processing
(This step can be skipped for using `./data/maskseq1by1_0.2.csv` directly)
```bash
python ./finetune/ankh_ft_data_1by1.py
```
#### Run
```bash
python ./finetune/ankh_ft_noval.py
```

### 3) AMPainter framework
#### Train agent network
```bash
python ./RLloop/reinforce.py
```
#### Evolve input sequences (fine-tuning mode, suggested)
Default params: `lr = 1e-3`, `batch_size = 128`, `n_steps = 10`, `iterations = 8`
```bash
python ./RLevolve/reinforce.py
```
#### Evolve input sequences (parallel mode)
```bash
python ./RLevolve/reinforce_para.py
```
