from dataloader import create_data_loaders
from models.transolver_model import Model as Transolver_Model
from models.Transolver_seg import Model as Transolver_SEG_Model
from models.Transolver_seg_v2 import Model as Transolver_SEG_V2_Model
from models.SegLinearNO import Model as SegLinearNO
from models.SegLinearNO_v2 import Model as SegLinearNO_v2
from models.LinearAttnNeuralOperator import LinearAttentionNeuralOperator as LinearNO
from models.RopeLinearNO import LinearAttentionNeuralOperator as RopeLinearNO
from models.RopeLinearNO_v2 import LinearAttentionNeuralOperator as RopeLinearNO_v2
from models.LinearNO_se import LinearAttentionNeuralOperator as LinearNO_se

# from models.mlp import MLP as MLP_Model
# from models.figconv import FigConv_Model
# from models.figconv import Multi_grid_model

#from utils.ml import train, test
from utils.ml_SATO import train, test

import os
import torch
import argparse
import yaml
from pathlib import Path

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
parser.add_argument('--exp', type=str, default=config.get('exp', 'ShapeNet'))
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

save_path = Path(f"experiment_{args.exp}") / f"best_model_{model_flag}.pth"

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
    # 假设样本文件为 .pkl 格式，文件名（不含扩展名）即为索引
    sample_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pkl')]
    # 提取索引，假设文件名是纯数字
    ALL_INDEX = sorted([int(os.path.splitext(f)[0]) for f in sample_files
                        if os.path.splitext(f)[0].isdigit()])

    num_samples = len(ALL_INDEX)
    print(f"Total samples found in {DATA_PATH}: {num_samples}")

    TRAIN_index, VAL_index, TEST_index = [], [], []

    for i, idx in enumerate(ALL_INDEX):
        r = i % 10
        if r < 6:        # 0–5 → 6/10 → train
            TRAIN_index.append(idx)
        elif r < 9:      # 6–8 → 3/10 → val
            VAL_index.append(idx)
        else:            # 9 → 1/10 → test
            TEST_index.append(idx)

    #Debug
    #TRAIN_index = config.get('train_index', [1, 2, 3, 4])
    #VAL_index = config.get('val_index', [5, 6, 7, 8])
    #TEST_index = config.get('test_index', [9, 10])
    print(f"Train: {len(TRAIN_index)}, Val: {len(VAL_index)}, Test: {len(TEST_index)}")

    coordinate_only_models = {
        'transolver',
        'LinearNO',
        'RopeLinearNO',
        'RopeLinearNO_v2',
    }
    train_loader, val_loader, test_loader, normalization_scalars = create_data_loaders(
        DATA_PATH,
        batch_size=args.batch_size,
        train_index=TRAIN_index,
        val_index=VAL_index,
        test_index=TEST_index,
        shuffle=args.shuffle,
        predicted_feature_name=predicted_feature_name,
        coordinate_only=model_name in coordinate_only_models,
    )
    if normalization_scalars is None:
        raise FileNotFoundError(
            f"normalization_scalars.pkl was not found in {DATA_PATH}. "
            "Run preprocess/process_data.py before training."
        )

    # Create model
    print("\n🏗️  Creating model...")
    if model_name == 'transolver':
        # Improved hyperparameters for better performance
        model = Transolver_Model(
            space_dim=args.space_dim,
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
            space_dim=args.space_dim,
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
            space_dim=args.space_dim,
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
            space_dim=args.space_dim,
            out_dim=1,
            n_layers=8,        # Increased from 8 to 12
            n_hidden=256,       # Increased from 256 to 512
            dropout=0.0,        # Added dropout for regularization
            n_head=8,
            act='gelu',
            mlp_ratio=2
        )
    elif model_name == 'SegLinearNO_v2':
        model = SegLinearNO_v2(
            space_dim=args.space_dim,
            out_dim=1,
            n_layers=8,        # Increased from 8 to 12
            n_hidden=256,       # Increased from 256 to 512
            dropout=0.0,        # Added dropout for regularization
            n_head=8,
            act='gelu',
            mlp_ratio=2
        )
    elif model_name == 'LinearNO':
        model = LinearNO(
            space_dim=args.space_dim,
            n_layers=8,
            n_hidden=256,
            n_head=8,
            Time_Input=False,
            mlp_ratio=2,fun_dim=0,out_dim=1,key_ratio=1,
            unified_pos=False,isregular=False)
    elif model_name == 'RopeLinearNO':
        model = RopeLinearNO(
            space_dim=args.space_dim,
            n_layers=8,
            n_hidden=256,
            n_head=8,
            Time_Input=False,
            mlp_ratio=2,fun_dim=0,out_dim=1,key_ratio=1,
            unified_pos=False,isregular=False)
    elif model_name == 'RopeLinearNO_v2':
        model = RopeLinearNO_v2(
            space_dim=args.space_dim,
            n_layers=8,
            n_hidden=256,
            n_head=8,
            Time_Input=False,
            mlp_ratio=2, fun_dim=0, out_dim=1, key_ratio=1,
            unified_pos=False, isregular=False,
        )
    elif model_name == 'LinearNO_se':
        model = LinearNO_se(
            space_dim=args.space_dim,
            n_layers=8,
            n_hidden=256,
            n_head=8,
            Time_Input=False,
            mlp_ratio=2, fun_dim=0, out_dim=1, key_ratio=1,
            unified_pos=False, isregular=False,
            patch_size=args.patch_size,
            shift=args.shift,
        )
    else:
        raise ValueError(f"Model name {model_name} not supported")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # load the pre-trained model if it exists
    if phase != 'restart_train' and save_path.exists():
        try:
            model.load_state_dict(
                torch.load(
                    save_path,
                    map_location=torch.device('cpu'),
                    weights_only=True,
                )
            )
            print(f"Loaded pre-trained model from {save_path}")
        except (RuntimeError, TypeError, ValueError) as error:
            print(f"No compatible pre-trained model found at {save_path}: {error}")

    # Training
    if phase == 'train' or phase == 'restart_train':
        print("\n🎯 Starting training...")
        history = train(
            model_name=model_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            normalization_scalars=normalization_scalars,
            args = args,
            num_epochs=num_epochs,        # Increased epochs
            eval_freq=args.eval_freq,
            save_path=save_path,
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
        model_path=save_path,
        predicted_feature_name=predicted_feature_name
    )
    print("\n✅ Pipeline completed!")


if __name__ == "__main__":
    main()
