import argparse
from pathlib import Path


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--dataset', type=str, default='AT')
    parser.add_argument('--nprocessors', type=int, default=24)
    parser.add_argument('--total_chunks', type=int, default=1)
    parser.add_argument('--chunk_id', type=int, default=0)
    parser.add_argument('--file_identifier', type=str)
    parser.add_argument('--save_type', type=str, default='dgl')
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args


def parse_training_config(add_pred_conf=False):
    parser = argparse.ArgumentParser('RetroRanker')
    parser.add_argument('--name', default='RetroRanker',
                        help='Name of the model')
    parser.add_argument('-d', '--dataset', default='',
                        help='Dataset to use (Training)')
    parser.add_argument('-b', '--batch_size', type=int, default=784 if add_pred_conf else 512,
                        help='Batch size of dataloader')
    parser.add_argument('--out_node_feats', default=512,
                        help='Output feature size for atoms')
    parser.add_argument('-n', '--num-epochs', type=int,
                        default=50, help='Maximum number of epochs for training')
    parser.add_argument('-p', '--patience', type=int,
                        default=30, help='Patience for early stopping')
    parser.add_argument('-cl', '--max-clip', type=int,
                        default=20, help='Maximum number of gradient clip')
    parser.add_argument('-lr', '--learning-rate', type=float,
                        default=3e-4, help='Learning rate of optimizer')
    parser.add_argument('-l2', '--weight-decay', type=float,
                        default=1e-5, help='Weight decay of optimizer')
    parser.add_argument('-ss', '--schedule_step', type=int,
                        default=10, help='Step size of learning scheduler')
    parser.add_argument('-nw', '--num-workers', type=int,
                        default=24, help='Number of processes for data loading')
    parser.add_argument('-pe', '--print-every', type=int, default=1000,
                        help='Print the training progress every X mini-batches')
    parser.add_argument('--root', type=str,
                        default='./')
    parser.add_argument('--exp_name', type=str)

    if add_pred_conf:
        parser.add_argument('--regen', action='store_true',
                            help='re-calculate the score file')
        parser.add_argument('-dt', '--testset', default='',
                            help='Dataset to use (Test)')
    args = parser.parse_args()

    if add_pred_conf:
        if not args.testset:
            args.testset = args.dataset
    return args


def get_number_of_total_chunks(model_type, dataset):
    if model_type=='gnn_dgl':
        return 5
    elif model_type == 'gh':
        return 30
    raise

def get_number_of_total_test_records(dataset):
    if dataset == 'R-SMILES':
        return 96023
    if dataset == 'AT':
        return 96870    
    raise
