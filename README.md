# Prompt Tuning
Codebase for Prompt Tuning experiments on T5. Includes:
* regular prompt tuning (following Lester et al)
* residual prompt tuning (our modification)
* full model tuning

<!-- To create nlp virtual env., run:
conda env create -f environment.yaml -->
## Installation
```bash
git clone https://github.com/arazd/soft_prompts
cd soft_prompts
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

## Repo structure
