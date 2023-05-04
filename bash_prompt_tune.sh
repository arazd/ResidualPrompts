#!/bin/bash
#SBATCH --job-name=prompts
#SBATCH --gres=gpu:1
#SBATCH --account=all
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/home/%u/prompt_tuning_log_%j.log

source ~/miniconda/bin/activate 
conda init
source activate nlp

HPARAMS=(
    "--save_name wsc_residual_prompts_10tokens_1 --task wsc --prefix_MLP MLP1"
    "--save_name wsc_residual_prompts_10tokens_2 --task wsc --prefix_MLP MLP1"
    "--save_name wsc_residual_prompts_10tokens_3 --task wsc --prefix_MLP MLP1"

    "--save_name wsc_prompt_tuning_10tokens_1 --task wsc"
    "--save_name wsc_prompt_tuning_10tokens_2 --task wsc"
    "--save_name wsc_prompt_tuning_10tokens_3 --task wsc"
)

cmd="python train.py ${HPARAMS[SLURM_ARRAY_TASK_ID]} \
    --lr 0.3 --freeze_weights 1 --freeze_except xxxx \
    --model_name t5-base --early_stopping 1 \
    --test_eval_after_every_task 1 --select_k_per_class -1 \
    --batch_size 8 --num_epochs 20 --prefix_len 10 \
    --save_dir /data/home/%u/residual_prompts_results/"

echo $cmd
eval $cmd
