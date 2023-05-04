#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --gres=gpu:1
#SBATCH --account=all
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/%u/residual_prompts/finetune_log_%j.log

source ~/miniconda/bin/activate 
conda init
source activate nlp

HPARAMS=(
    # "--save_name copa_FT_1 --task copa"
    # "--save_name copa_FT_2 --task copa"
    # "--save_name copa_FT_3 --task copa"

    # "--save_name cb_FT_1 --task cb"
    # "--save_name cb_FT_2 --task cb"
    # "--save_name cb_FT_3 --task cb"

    # "--save_name multirc_FT_1 --task multirc"
    # "--save_name multirc_FT_2 --task multirc"
    # "--save_name multirc_FT_3 --task multirc"

    # "--save_name record_FT_1 --task record"
    # "--save_name record_FT_2 --task record"
    # "--save_name record_FT_3 --task record"

    # "--save_name rte_superglue_FT_1 --task rte_superglue"
    # "--save_name rte_superglue_FT_2 --task rte_superglue"
    # "--save_name rte_superglue_FT_3 --task rte_superglue"

    # "--save_name wic_FT_1 --task wic"
    # "--save_name wic_FT_2 --task wic"
    # "--save_name wic_FT_3 --task wic"

    "--save_name wsc_finetune_full_1 --task wsc"
    "--save_name wsc_finetune_full_2 --task wsc"
    "--save_name wsc_finetune_full_3 --task wsc"

    # "--save_name boolq_FT_1 --task boolq"
    # "--save_name boolq_FT_2 --task boolq"
    # "--save_name boolq_FT_3 --task boolq"

)

cmd="python train.py ${HPARAMS[SLURM_ARRAY_TASK_ID]} \
    --lr 3e-4 --freeze_weights 0 --freeze_except xxxx --model_name t5-base --early_stopping 1  --prefix_len 0 \
    --test_eval_after_every_task 1 --select_k_per_class -1  --num_epochs 50 --batch_size 4 \
    --save_dir /data/home/%u/residual_prompts_results/"

echo $cmd
eval $cmd
