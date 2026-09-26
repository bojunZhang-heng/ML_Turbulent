from dataloader import create_data_loaders
from utils.metric import compute_relative_error
from models.transolver_model import Model as Transolver_Model
from models.Transolver_seg import Model as Transolver_SEG_Model
from models.Transolver_seg_v2 import Model as Transolver_SEG_V2_Model
from models.SegLinearNO import Model as SegLinearNO
# from models.mlp import MLP as MLP_Model
# from models.figconv import FigConv_Model
# from models.figconv import Multi_grid_model

from utils.ml import train, test

import os
import torch
import argparse
import pickle
import numpy as np
import yaml

# Load YAML defaults first. Explicit command-line arguments override them.
config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument('--config', type=str, default=None)
config_args, _ = config_parser.parse_known_args()
config = {}
if config_args.config:
    with open(config_args.config, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file) or {}

parser = argparse.ArgumentParser(parents=[config_parser])
parser.set_defaults(**config)
parser.add_argument('--model_name', type=str, default=config.get('model_name', "transolver"))
parser.add_argument('--predicted_feature_name', type=str, default=config.get('predicted_feature_name', "pressure"))
parser.add_argument('--phase', type=str, default=config.get('phase', "train"), choices=["train", "restart_train", "test"])
parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs', 500))
parser.add_argument('--eval_freq', type=int, default=config.get('eval_freq', 10))
parser.add_argument('--learning_rate', type=float, default=config.get('learning_rate', 2e-5))
parser.add_argument('--data_path', type=str, default=config.get('data_path', ""))
parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 1))
parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=config.get('shuffle', True))
args = parser.parse_args()

# set the experiment settings
model_name = args.model_name
predicted_feature_name = args.predicted_feature_name
phase = args.phase
model_flag = model_name + '_' + predicted_feature_name
num_epochs = args.num_epochs

def main():
    """
    Main function to run training and testing.
    """
    print("🚀 Starting VTK Data Processing Pipeline")

    # Create data loaders
    print("\n🔄 Creating data loaders...")

    DATA_PATH = args.data_path
    if not DATA_PATH.endswith(os.sep):
        DATA_PATH += os.sep

    # 动态扫描 DATA_PATH 下的样本文件，生成索引列表
    # 假设样本文件为 .vtk 格式，文件名（不含扩展名）即为索引
    sample_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pkl')]
    # 提取索引，假设文件名是纯数字
    ALL_INDEX = sorted([int(os.path.splitext(f)[0]) for f in sample_files
                        if os.path.splitext(f)[0].isdigit()])

    num_samples = len(ALL_INDEX)
    print(f"Total samples found in {DATA_PATH}: {num_samples}")

    TRAIN_index, VAL_index, TEST_index = [], [], []

    for i, idx in enumerate(ALL_INDEX):
        r = i % 10
        if r < 7:        # 0–6 → 7/10 → train
            TRAIN_index.append(idx)
        elif r == 7:     # 7 → 1/10 → val
            VAL_index.append(idx)
        else:            # 8–9 → 2/10 → test
            TEST_index.append(idx)

    #Debug
    #TRAIN_index = config.get('train_index', [1, 2, 3, 4])
    #VAL_index = config.get('val_index', [5, 6, 7, 8])
    #TEST_index = config.get('test_index', [9, 10])
    print(f"Train: {len(TRAIN_index)}, Val: {len(VAL_index)}, Test: {len(TEST_index)}")

    with open(DATA_PATH + "normalization_scalars.pkl", "rb") as f:
        normalization_scalars = pickle.load(f)
    train_loader, val_loader, test_loader, _ = create_data_loaders(
        DATA_PATH,
        batch_size=args.batch_size,
        train_index=TRAIN_index,
        val_index=VAL_index,
        test_index=TEST_index,
        shuffle=args.shuffle,
        predicted_feature_name=predicted_feature_name
    )

    # Create model
    print("\n🏗️  Creating model...")
    if model_name == 'transolver':
        # Improved hyperparameters for better performance
        model = Transolver_Model(
            space_dim=6,
            out_dim=1,
            n_layers=8,        # Increased from 8 to 12
            n_hidden=256,       # Increased from 256 to 512
            dropout=0.0,        # Added dropout for regularization
            n_head=8,
            act='gelu',
            mlp_ratio=2,        # Increased from 2 to 4
            slice_num=64
        )
    elif model_name == 'transolver_seg':
        model = Transolver_SEG_Model(
            space_dim=6,
            out_dim=1,
            n_layers=8,        # Increased from 8 to 12
            n_hidden=256,       # Increased from 256 to 512
            dropout=0.0,        # Added dropout for regularization
            n_head=8,
            act='gelu',
            mlp_ratio=2,        # Increased from 2 to 4
        )  # Changed from 3 to 6 for 6D features
    elif model_name == 'transolver_seg_v2':
        model = Transolver_SEG_V2_Model(
            space_dim=6,
            out_dim=1,
            n_layers=8,        # Increased from 8 to 12
            n_hidden=256,       # Increased from 256 to 512
            dropout=0.0,        # Added dropout for regularization
            n_head=8,
            act='gelu',
            mlp_ratio=2
        )
    elif model_name == 'SegLinearNO':
        model = SegLinearNO(
            space_dim=6,
            out_dim=1,
            n_layers=8,        # Increased from 8 to 12
            n_hidden=256,       # Increased from 256 to 512
            dropout=0.0,        # Added dropout for regularization
            n_head=8,
            act='gelu',
            mlp_ratio=2
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # load the pre-trained model if it exists
    if phase != 'restart_train':
        try:
            if os.path.exists("trained_models/best_model_{}.pth".format(model_flag)):
                # weights_only=False is safe here since we're loading our own trained models
                model.load_state_dict(
                    torch.load(
                        "trained_models/best_model_{}.pth".format(model_flag),
                        map_location=torch.device('cpu'),
                        weights_only=False
                    )
                )
                print("Loaded pre-trained model")
        except:
            print("No compatible pre-trained model found")

    # Training
    if phase == 'train' or phase == 'restart_train':
        print("\n🎯 Starting training...")
        history = train(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            normalization_scalars=normalization_scalars,
            num_epochs=num_epochs,        # Increased epochs
            learning_rate=args.learning_rate,    # Increased learning rate for AdamW
            eval_freq=args.eval_freq,
            save_path="trained_models/best_model_{}.pth".format(model_flag),
            predicted_feature_name=predicted_feature_name
        )

    # Testing
    print("\n🧪 Starting testing...")
    print("\n Testing on test dataset...")
    _ = test(
        model_name=model_name,
        model=model,
        test_loader=test_loader,
        normalization_scalars=normalization_scalars,
        model_path="trained_models/best_model_{}.pth".format(model_flag),
        predicted_feature_name=predicted_feature_name
    )
    print("\n✅ Pipeline completed!")


if __name__ == "__main__":
    main()
