import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import logging, os, argparse

from prompt_tuner2 import PromptTuner


def main(args):
    save_path = os.path.join(args.save_dir, args.save_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    prompt_tuner = PromptTuner(model_name=args.model_name,
                               task=args.task,
                               batch_size=args.batch_size,
                               select_k_per_class=args.select_k_per_class,
                               prefix_len=args.prefix_len,
                               prefix_path=args.prefix_path if args.prefix_path!='' else None, # path to the pre-trained progressive prompt
                               freeze_weights=args.freeze_weights==1,
                               freeze_except=args.freeze_except,
                               lr=args.lr,
                               weight_decay=1e-5,
                               seq_len=args.seq_len,
                               early_stopping=args.early_stopping==1,
                               random_prompt_init=args.random_prompt_init==1,
                               prefix_MLP=args.prefix_MLP,
                               residual=args.residual==1,
                               bottleneck_size=args.bottleneck_size, # bottleneck size in case of using MLP reparametrization
                               mlp_dropout=args.mlp_dropout,
                               nonlinearity=args.nonlinearity,
                               mlp_lr=None,
                               separate_mlps=args.separate_mlps==1,
                               mlp_layer_norm=args.mlp_layer_norm==1,
                               weight_decay_mlp=None,
                               get_test_subset=args.get_test_subset,
                              )


    if args.get_test_subset==0:
        print("Not creating test subset")

    if args.num_epochs<=50:
        eval_every_N = 1
    elif args.num_epochs>50 and args.num_epochs<=200:
        eval_every_N = 5
    elif args.num_epochs>200:
        eval_every_N = 25 #10

    score_dict = prompt_tuner.train_one_task(epochs=args.num_epochs,
                                             eval_every_N=eval_every_N,
                                             save_path=save_path)
    np.save(os.path.join(save_path, 'score_dict.npy'), score_dict)
    if prompt_tuner.model.prompt!=None:
        np.save(os.path.join(save_path, 'prompt.npy'), prompt_tuner.model.prompt.detach().cpu().numpy())




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description='NLP training script in PyTorch'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        help='base directory of all models / features (should not be changed)',
        default='/data/home/arazdai/residual_prompts/results/' #'/scratch/hdd001/home/anastasia/CL/'
    )

    parser.add_argument(
        '--save_name',
        type=str,
        help='folder name to save',
        required=True
    )

    # parser.add_argument(
    #     '--task_list',
    #     nargs='+',
    #     help='List of tasks for training',
    #     required=True
    # )

    parser.add_argument(
        '--task',
        type=str,
        help='Task to train on',
        required=True
    )


    parser.add_argument(
        '--model_name',
        type=str,
        help='Name of the model used for training',
        default="t5-base"
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of epochs to train model',
        default=5
    )


    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size',
        default=8
    )

    parser.add_argument(
        '--seq_len',
        type=int,
        help='Length of a single repeat (in #tokens)',
        default=512
    )

    parser.add_argument(
        '--prefix_len',
        type=int,
        help='Length of prompt (in #tokens)',
        default=10
    )

    parser.add_argument(
        '--prefix_path',
        type=str,
        help='path to a pre-trained progressive prefix (for superGLUE experiments)',
        default=''
    )


    parser.add_argument(
        '--lr',
        type=float,
        help='Learning rate',
        default=0.3
    )

    parser.add_argument(
        '--mlp_dropout',
        type=float,
        help='Dropout for MLP reparametrization network',
        default=0.0
    )

    parser.add_argument(
        '--limit_val_set',
        type=int,
        help='Limit validation set to N per class (to speed up experiments)',
        default=-1
    )

    parser.add_argument(
        '--select_k_per_class',
        type=int,
        help='Select k examples from each class (default -1, i.e. no changes to the original dataset)',
        default=-1
    )

    parser.add_argument(
        '--test_eval_after_every_task',
        type=int,
        help='Whether to re-evaluate test accuracy after every task (0 - False, 1 - True)',
        default=0
    )

    parser.add_argument(
        '--freeze_weights',
        type=int,
        help='Whether to freeze model weigts (except word emb)',
        default=0
    )

    parser.add_argument(
        '--freeze_except',
        type=str,
        help='If freeze_weights==1, freeze all weights except those that contain this keyword',
        default='xxxxxxx' # freeze all
    )

    parser.add_argument(
        '--get_test_subset',
        type=int,
        help='Whether to create a separate test split',
        default=0
    )

    parser.add_argument(
        '--early_stopping',
        type=int,
        help='If early_stopping==1, do early stopping based on val accuracy',
        default=0 # freeze all
    )

    parser.add_argument(
        '--random_prompt_init',
        type=int,
        help='Init prompt from random uniform',
        default=0
    )

    parser.add_argument(
        '--prefix_MLP',
        type=str,
        help='Type of MLP reparametrization (if None - use Lester original implementation)',
        default='None' # freeze all
    )

    parser.add_argument(
        '--residual',
        type=int,
        help='Use residual MLP',
        default=1 # add residual connection to MLP
    )

    parser.add_argument(
        '--separate_mlps',
        type=int,
        help='Use separate MLP for each prompt token',
        default=0 # add residual connection to MLP
    )

    parser.add_argument(
        '--mlp_layer_norm',
        type=int,
        help='Do layer norm in MLP',
        default=1 # use layer norm
    )

    parser.add_argument(
        '--nonlinearity',
        type=str,
        help='Type of MLP nonlinearity',
        default='relu' # freeze all
    )

    parser.add_argument(
        '--bottleneck_size',
        type=int,
        help='MLP bottleneck size',
        default=800
    )

    main(parser.parse_args())
