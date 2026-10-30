from dataloader import create_data_loaders, discover_multi_vehicle_samples
from models.transolver_model import Model as Transolver_Model

from models.LinearAttnNeuralOperator import LinearAttentionNeuralOperator as LinearNO
from models.CANO import LinearAttentionNeuralOperator as CANO
from models.RETO_plus import Model as RETO_plus
from models.SATO import Model as SATO
from datetime import datetime

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


CONFIG_SECTIONS = (
    "experiment",
    "model",
    "attention",
    "position_encoding",
    "serialization",
    "checkpoint",
    "training",
    "data",
)


def flatten_config(raw_config):
    """Support categorized YAML while remaining compatible with flat files."""
    flat = {key: value for key, value in raw_config.items() if not isinstance(value, dict)}
    for section in CONFIG_SECTIONS:
        values = raw_config.get(section, {})
        if values is None:
            continue
        if not isinstance(values, dict):
            raise TypeError(f"YAML section {section!r} must be a mapping")
        flat.update(values)
    return flat


grouped_config = config
config = flatten_config(grouped_config)

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
parser.add_argument('--dataset_root', type=str, default=config.get('dataset_root', ""))
parser.add_argument('--selection_mode', type=str, choices=['first', 'random'], default=config.get('selection_mode', 'random'))
parser.add_argument('--selection_seed', type=int, default=config.get('selection_seed', 42))
parser.add_argument('--train_ratio', type=float, default=config.get('train_ratio', 0.6))
parser.add_argument('--val_ratio', type=float, default=config.get('val_ratio', 0.3))
parser.add_argument('--test_ratio', type=float, default=config.get('test_ratio', 0.1))
parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 1))
parser.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=config.get('shuffle', True))
parser.add_argument('--space_dim', type=int, default=config.get('space_dim', 3))
parser.add_argument('--out_dim', type=int, default=config.get('out_dim', 1))
parser.add_argument('--n_layers', type=int, default=config.get('n_layers', 8))
parser.add_argument('--n_hidden', type=int, default=config.get('n_hidden', 256))
parser.add_argument('--dropout', type=float, default=config.get('dropout', 0.0))
parser.add_argument('--n_head', type=int, default=config.get('n_head', 8))
parser.add_argument('--act', type=str, default=config.get('act', 'gelu'))
parser.add_argument('--mlp_ratio', type=float, default=config.get('mlp_ratio', 2.0))
parser.add_argument('--fun_dim', type=int, default=config.get('fun_dim', 0))
parser.add_argument('--key_ratio', type=int, default=config.get('key_ratio', 1))
parser.add_argument('--slice_num', type=int, default=config.get('slice_num', 64))
parser.add_argument('--ref', type=int, default=config.get('ref', 8))
parser.add_argument('--H', type=int, default=config.get('H', 85))
parser.add_argument('--W', type=int, default=config.get('W', 85))
parser.add_argument('--unified_pos', action=argparse.BooleanOptionalAction, default=config.get('unified_pos', False))
parser.add_argument('--isregular', action=argparse.BooleanOptionalAction, default=config.get('isregular', False))
parser.add_argument('--time_input', action=argparse.BooleanOptionalAction, default=config.get('time_input', False))
parser.add_argument('--patch_size', type=int, default=config.get('patch_size', 20))
parser.add_argument('--shift', type=int, default=config.get('shift', 2))
parser.add_argument('--n_iter', type=int, default=config.get('n_iter', 1))
parser.add_argument('--num_position_frequencies', type=int, default=config.get('num_position_frequencies', 6))
parser.add_argument('--initial_relative_gate', type=float, default=config.get('initial_relative_gate', 0.05))
parser.add_argument('--checkpoint_relative_terms', action=argparse.BooleanOptionalAction, default=config.get('checkpoint_relative_terms', True))
parser.add_argument('--cluster_num', type=int, default=config.get('cluster_num', 64))
parser.add_argument('--cluster_temperature_init', type=float, default=config.get('cluster_temperature_init', 1.0))
parser.add_argument('--checkpoint_segment_size', type=int, default=config.get('checkpoint_segment_size', 2))
parser.add_argument('--checkpoint_num_blocks', type=int, default=config.get('checkpoint_num_blocks', None))
parser.add_argument('--layer_scale_init', type=float, default=config.get('layer_scale_init', 0.1))
parser.add_argument('--initial_rope_gate', type=float, default=config.get('initial_rope_gate', 0.05))
args = parser.parse_args()

# set the experiment settings
model_name = args.model_name
predicted_feature_name = args.predicted_feature_name
phase = args.phase
num_epochs = args.num_epochs

ts = datetime.now().strftime('%Y%m%d_%H%M')  # 整个实验共用一个时间戳
model_flag = f"{model_name}_{predicted_feature_name}_{ts}"

save_path = Path(f"experiment_{args.exp}") / f"best_model_{model_flag}.pth"

def main():
    """
    Main function to run training and testing.
    """
    print("🚀 Starting VTK Data Processing Pipeline")
    if config_args.config:
        print(f"\n📋 Configuration: {config_args.config}")
        print(yaml.safe_dump(grouped_config, sort_keys=False, allow_unicode=True))

    # Create data loaders
    print("\n🔄 Creating data loaders...")

    data_config = grouped_config.get("data", {})
    vehicle_configs = data_config.get("vehicle_datasets")
    if vehicle_configs:
        DATA_SOURCE, TRAIN_index, VAL_index, TEST_index, vehicle_summary = (
            discover_multi_vehicle_samples(
                vehicle_configs=vehicle_configs,
                data_root=args.dataset_root or None,
                selection_mode=args.selection_mode,
                selection_seed=args.selection_seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
            )
        )
        print("\nVehicle sampling summary:")
        for vehicle, values in vehicle_summary.items():
            print(
                f"  {vehicle}: available={values['available']}, "
                f"selected={values['selected']}, train={values['train']}, "
                f"val={values['val']}, test={values['test']}"
            )
        print(
            f"Combined: train={len(TRAIN_index)}, val={len(VAL_index)}, "
            f"test={len(TEST_index)}"
        )
    else:
        DATA_PATH = args.data_path
        if not DATA_PATH:
            raise ValueError(
                "Configure data.vehicle_datasets or provide data.data_path"
            )
        if not DATA_PATH.endswith(os.sep):
            DATA_PATH += os.sep
        sample_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pkl')]
        ALL_INDEX = sorted(
            int(os.path.splitext(file)[0])
            for file in sample_files
            if os.path.splitext(file)[0].isdigit()
        )
        print(f"Total samples found in {DATA_PATH}: {len(ALL_INDEX)}")
        TRAIN_index, VAL_index, TEST_index = [], [], []
        for position, sample_id in enumerate(ALL_INDEX):
            ratio_position = position % 10
            if ratio_position < 6:
                TRAIN_index.append(sample_id)
            elif ratio_position < 9:
                VAL_index.append(sample_id)
            else:
                TEST_index.append(sample_id)
        DATA_SOURCE = DATA_PATH
        print(
            f"Train: {len(TRAIN_index)}, Val: {len(VAL_index)}, "
            f"Test: {len(TEST_index)}"
        )

    coordinate_only_models = {
        'Transolver',
        'LinearNO',
        'CANO',
        'RETO_plus',
    }
    train_loader, val_loader, test_loader, normalization_scalars = create_data_loaders(
        DATA_SOURCE,
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
            "normalization_scalars.pkl was not found for the selected data. "
            "Run preprocess/process_data.py before training."
        )

    # Create model
    print("\n🏗️  Creating model...")
    if model_name == 'Transolver':
        model = Transolver_Model(
            space_dim=args.space_dim,
            out_dim=args.out_dim,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            dropout=args.dropout,
            n_head=args.n_head,
            act=args.act,
            mlp_ratio=args.mlp_ratio,
            fun_dim=args.fun_dim,
            slice_num=args.slice_num,
            ref=args.ref,
            unified_pos=args.unified_pos,
            checkpoint_segment_size=args.checkpoint_segment_size,
            checkpoint_num_blocks=args.checkpoint_num_blocks,
        )
    elif model_name == 'LinearNO':
        model = LinearNO(
            space_dim=args.space_dim,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            dropout=args.dropout,
            n_head=args.n_head,
            Time_Input=args.time_input,
            act=args.act,
            mlp_ratio=args.mlp_ratio,
            fun_dim=args.fun_dim,
            out_dim=args.out_dim,
            key_ratio=args.key_ratio,
            ref=args.ref,
            unified_pos=args.unified_pos,
            H=args.H,
            W=args.W,
            isregular=args.isregular,
        )
    elif model_name == 'CANO':
        model = CANO(
            space_dim=args.space_dim,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            dropout=args.dropout,
            n_head=args.n_head,
            Time_Input=args.time_input,
            act=args.act,
            mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim,
            out_dim=args.out_dim, key_ratio=args.key_ratio,
            ref=args.ref, unified_pos=args.unified_pos,
            H=args.H, W=args.W, isregular=args.isregular,
            cluster_num=args.cluster_num,
            checkpoint_segment_size=args.checkpoint_segment_size,
            checkpoint_num_blocks=args.checkpoint_num_blocks,
            layer_scale_init=args.layer_scale_init,
            cluster_temperature_init=args.cluster_temperature_init,
        )
    elif model_name == 'RETO_plus':
        model = RETO_plus(
            space_dim=args.space_dim,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            dropout=args.dropout,
            n_head=args.n_head,
            Time_Input=args.time_input,
            act=args.act,
            mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim,
            out_dim=args.out_dim, key_ratio=args.key_ratio,
            ref=args.ref, unified_pos=args.unified_pos,
            isregular=args.isregular,
            cluster_num=args.cluster_num,
            num_position_frequencies=args.num_position_frequencies,
            checkpoint_segment_size=args.checkpoint_segment_size,
            checkpoint_num_blocks=args.checkpoint_num_blocks,
            layer_scale_init=args.layer_scale_init,
            cluster_temperature_init=args.cluster_temperature_init,
        )
    elif model_name == 'SATO':
        model = SATO(
            space_dim=args.space_dim,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            dropout=args.dropout,
            n_head=args.n_head,
            act=args.act,
            slice_num=args.slice_num,
            mlp_ratio=args.mlp_ratio, fun_dim=args.fun_dim,
            patch_size=args.patch_size,
            shift=args.shift,
            n_iter=args.n_iter,
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
