# Residual Prompt Tuning
This repository contains the original implementation for ["***Residual Prompt Tuning: Improving Prompt Tuning
with Residual Reparameterization***"](https://arxiv.org/abs/2305.03937) (ACL 2023) by Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Madian Khabsa, Mike Lewis, Jimmy Ba and Amjad Almahairi.

<!-- ![Residual Prompt Tuning illustration](/images/residual_pt_method.png) -->
<img src="images/residual_pt_method.png" align="right" width="300">
<!-- **Illustration of Residual Prompt Tuning and comparison with prompt tuning by Lester et al. (2021). ** -->

🎊 **Our work is accepted to ACL Findings 2023!** 

<!-- Our paper here - ["Residual Prompt Tuning: Improving Prompt Tuning
with Residual Reparameterization"](https://arxiv.org/abs/2305.03937), Findings of ACL 2023. -->

### Table of contents
* [Overview](#Overview)
* [Installation](#Installation)
* [Training](#Training) 
<!-- * [How to cite](#raising_hand-questions) -->


## Overview
#### What are Residual Prompts?
We introduce *Residual Prompt Tuning* – a simple and efficient method that significantly improves the performance and stability of prompt tuning. We propose to reparameterize soft prompt embedings using a shallow network with a residual connection. 

#### Intuition behind prompt reparameterization
This reparameterization gives the model more flexibility to decide between using a separate embedding for each prompt token versus the representation obtained from the
shared reparameterization network. After training is completed, the reparameterization network can be discarded and original prompt embeddings can be replaced with their projections.

#### Codebase overview
Our codebase includes pytorch implementation of:
* original prompt tuning (following Lester et al.)
* residual prompt tuning (our modification)
* full model tuning

<!-- To create nlp virtual env., run:
conda env create -f environment.yaml -->
## Installation
Clone this repo as follows:
```bash
git clone https://github.com/arazd/ResidualPrompts
cd ResidualPrompts
conda env create -f environment.yaml
conda activate nlp
```

## Training
An example of training a 10-token soft prompt on WSC task using T5-base model and residual reparametrization with MLP1 type:
```bash
python train.py --task wsc --prefix_MLP MLP1 \
    --lr 0.3 --freeze_weights 1 --freeze_except xxxx \
    --model_name t5-base --early_stopping 1 \
    --test_eval_after_every_task 1 --select_k_per_class -1 \
    --batch_size 8 --num_epochs 20 --prefix_len 10 \
    --save_dir /home/%u/my_dir/ --save_name my_model_folder
```

<!-- ## Repo structure -->
## Reference
If you use the code for your work, please consider citing our paper:
```bibtex
@inproceedings{razdaibiedina2023residual,
   title={Residual Prompt Tuning: Improving Prompt Tuning with Residual Reparameterization},
   author={Razdaibiedina, Anastasia and Mao, Yuning and Hou, Rui and Khabsa, Madian and Lewis, Mike and Ba, Jimmy and Almahairi, Amjad},
   booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
   year={2023}
}
```
