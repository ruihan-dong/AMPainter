# AMPainter
a computational framework to evolve antimicrobial peptides via reinforcement learning

## Overview
This framework consists of three modules:
* a policy network to assign the mutation sites
* a fine-tuned protein language model to replace the assigned residues
* a predictor named HyperAMP to evaluate the antimicrobial activity

## Requirements
* pytorch
* DHG
* Ankh (and transformers)
* modlamp

## Usage
```python
python ./RLloop/reinforce.py
```
