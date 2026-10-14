import argparse
import json
import os
import random
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parent
SHAPENET_ROOT = REPO_ROOT / "ShapeNetCar"
if str(SHAPENET_ROOT) not in sys.path:
    sys.path.insert(0, str(SHAPENET_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate LinearNO on ShapeNetCar")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    return parser.parse_args()


def read_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("The YAML root must be a mapping.")
    return config


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def select_device(name):
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {name!r} was requested, but CUDA is unavailable.")
    return device


def safe_extract(archive, destination):
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (destination / member.name).resolve()
            if destination != member_path and destination not in member_path.parents:
                raise RuntimeError(f"Unsafe path {member.name!r} in {archive}")
        tar.extractall(destination)


def prepare_archives(data_dir, auto_extract):
    data_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for fold_id in range(9):
        fold_dir = data_dir / f"param{fold_id}"
        if fold_dir.is_dir():
            continue
        archive = data_dir / f"param{fold_id}.tar.gz"
        if auto_extract and archive.is_file():
            print(f"Extracting {archive.name} ...")
            safe_extract(archive, data_dir)
        else:
            missing.append(str(fold_dir))
    if missing:
        raise FileNotFoundError(
            "ShapeNetCar folds are missing. Set dataset.auto_extract: true or extract the archives first:\n"
            + "\n".join(missing)
        )


def cache_is_complete(samples, cache_dir):
    required = ("x.npy", "y.npy", "pos.npy", "surf.npy", "edge_index.npy")
    return all(all((cache_dir / sample / name).is_file() for name in required) for sample in samples)


def load_datasets(config):
    from dataset.dataset import GraphDataset
    from dataset.load_dataset import get_samples, load_train_val_fold_file

    dataset_config = config["dataset"]
    data_root = resolve_path(dataset_config["root"])
    data_dir = data_root / dataset_config.get("training_subdir", "training_data")
    cache_dir = data_root / dataset_config.get("cache_subdir", "preprocessed_data")
    prepare_archives(data_dir, dataset_config.get("auto_extract", True))

    fold_id = int(dataset_config.get("fold_id", 0))
    if not 0 <= fold_id < 9:
        raise ValueError("dataset.fold_id must be between 0 and 8.")

    samples_by_fold = get_samples(str(data_dir))
    all_samples = [sample for fold in samples_by_fold for sample in fold]
    if not all_samples:
        raise RuntimeError(f"No ShapeNetCar samples were found under {data_dir}")

    preprocessed = dataset_config.get("preprocessed", "auto")
    if preprocessed == "auto":
        preprocessed = cache_is_complete(all_samples, cache_dir)
    elif not isinstance(preprocessed, bool):
        raise ValueError("dataset.preprocessed must be true, false, or 'auto'.")
    if not preprocessed:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Preprocessed cache is incomplete; building it in {cache_dir}")

    loader_args = SimpleNamespace(
        data_dir=str(data_dir), save_dir=str(cache_dir), fold_id=fold_id
    )
    train_data, val_data, coef_norm, val_samples = load_train_val_fold_file(
        loader_args, preprocessed=preprocessed
    )
    if not train_data or not val_data:
        raise RuntimeError(
            "The training or validation split is empty. Remove an incomplete cache directory "
            "or set dataset.preprocessed: false to rebuild it."
        )

    graph_config = dataset_config.get("graph", {})
    use_cfd_mesh = bool(graph_config.get("use_cfd_mesh", True))
    radius = float(graph_config.get("radius", 0.2))
    train_dataset = GraphDataset(train_data, use_cfd_mesh=use_cfd_mesh, r=radius)
    val_dataset = GraphDataset(val_data, use_cfd_mesh=use_cfd_mesh, r=radius)
    return train_dataset, val_dataset, coef_norm, val_samples, data_dir


def build_model(config):
    from models.LinearAttnNeuralOperator import LinearAttentionNeuralOperator

    model_config = dict(config.get("model", {}))
    return LinearAttentionNeuralOperator(**model_config)


def train_model(config, device, train_dataset, val_dataset, coef_norm):
    import train as train_module

    training_config = config["training"]
    batch_size = int(training_config.get("batch_size", 1))
    epochs = int(training_config.get("epochs", 200))
    if batch_size != 1:
        raise ValueError("This ShapeNetCar implementation only supports training.batch_size: 1.")
    if epochs < 1:
        raise ValueError("training.epochs must be positive.")

    output_dir = resolve_path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    hparams = {
        "lr": float(training_config.get("learning_rate", 0.001)),
        "batch_size": batch_size,
        "nb_epochs": epochs,
    }
    model = build_model(config)
    print(model)
    print(f"Using device: {device}")
    return train_module.main(
        device,
        train_dataset,
        val_dataset,
        model,
        hparams,
        str(output_dir),
        val_iter=int(training_config.get("validation_interval", 10)),
        reg=float(training_config.get("pressure_loss_weight", 0.5)),
        coef_norm=coef_norm,
    )


def checkpoint_path(config):
    configured = config.get("evaluation", {}).get("checkpoint")
    if configured:
        return resolve_path(configured)
    epochs = int(config["training"].get("epochs", 200))
    return resolve_path(config["output"]["directory"]) / f"model_{epochs}.pth"


def load_model(config, device):
    path = checkpoint_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate_model(config, device, model, val_dataset, coef_norm, val_samples, data_dir):
    from scipy import stats
    from torch import nn
    from torch_geometric.loader import DataLoader
    from utils.drag_coefficient import cal_coefficient

    evaluation_config = config.get("evaluation", {})
    results_dir = resolve_path(evaluation_config.get("results_directory", "results/shapenet"))
    results_dir.mkdir(parents=True, exist_ok=True)
    save_predictions = bool(evaluation_config.get("save_predictions", True))
    compute_drag = bool(evaluation_config.get("compute_drag", True))

    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    l2_pressure, l2_velocity = [], []
    mse_pressure, mse_velocity = [], []
    true_drag, predicted_drag = [], []

    mean = torch.as_tensor(coef_norm[2], device=device)
    std = torch.as_tensor(coef_norm[3], device=device)
    for index, ((cfd_data, geometry), sample) in enumerate(
        zip(DataLoader(val_dataset, batch_size=1), val_samples)
    ):
        print(f"Evaluating {sample}")
        cfd_data, geometry = cfd_data.to(device), geometry.to(device)
        prediction = model((cfd_data, geometry))
        target = cfd_data.y
        prediction_denorm = prediction * std + mean
        target_denorm = target * std + mean

        surface = cfd_data.surf
        predicted_pressure = prediction_denorm[surface, -1]
        true_pressure = target_denorm[surface, -1]
        predicted_velocity = prediction_denorm[~surface, :-1]
        true_velocity = target_denorm[~surface, :-1]
        predicted_surface_velocity = prediction_denorm[surface, :-1]
        true_surface_velocity = target_denorm[surface, :-1]

        l2_pressure.append((torch.norm(predicted_pressure - true_pressure) / torch.norm(true_pressure)).item())
        l2_velocity.append((torch.norm(predicted_velocity - true_velocity) / torch.norm(true_velocity)).item())
        mse_pressure.append(criterion(prediction[surface, -1], target[surface, -1]).mean().item())
        mse_velocity.append(criterion(prediction[~surface, :-1], target[~surface, :-1]).mean(dim=0).cpu().numpy())

        if save_predictions:
            np.save(results_dir / f"{index}_pred.npy", prediction_denorm.cpu().numpy())
            np.save(results_dir / f"{index}_gt.npy", target_denorm.cpu().numpy())
        if compute_drag:
            predicted_drag.append(
                cal_coefficient(sample, predicted_pressure[:, None].cpu().numpy(), predicted_surface_velocity.cpu().numpy(), root=str(data_dir))
            )
            true_drag.append(
                cal_coefficient(sample, true_pressure[:, None].cpu().numpy(), true_surface_velocity.cpu().numpy(), root=str(data_dir))
            )

    velocity_rmse_components = np.sqrt(np.mean(mse_velocity, axis=0)) * np.asarray(coef_norm[3][:-1])
    metrics = {
        "relative_l2_pressure": float(np.mean(l2_pressure)),
        "relative_l2_velocity": float(np.mean(l2_velocity)),
        "pressure_rmse": float(np.sqrt(np.mean(mse_pressure)) * coef_norm[3][-1]),
        "velocity_rmse_components": velocity_rmse_components.tolist(),
        "velocity_rmse": float(np.sqrt(np.mean(np.square(velocity_rmse_components)))),
    }
    if compute_drag:
        true_drag = np.asarray(true_drag)
        predicted_drag = np.asarray(predicted_drag)
        metrics["drag_relative_error"] = float(np.mean(np.abs(predicted_drag - true_drag) / np.abs(true_drag)))
        metrics["drag_spearman_rho"] = float(stats.spearmanr(true_drag, predicted_drag).statistic)

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Evaluation results saved to {results_dir}")
    return metrics


def main():
    args = parse_args()
    config = read_config(args.config)
    mode = config.get("mode", "train_and_evaluate")
    valid_modes = {"prepare", "train", "evaluate", "train_and_evaluate"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {sorted(valid_modes)}")

    runtime_config = config.get("runtime", {})
    set_seed(int(runtime_config.get("seed", 0)), bool(runtime_config.get("deterministic", True)))
    device = select_device(runtime_config.get("device", "auto"))
    train_dataset, val_dataset, coef_norm, val_samples, data_dir = load_datasets(config)
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples")
    if mode == "prepare":
        return

    model = None
    if mode in {"train", "train_and_evaluate"}:
        model = train_model(config, device, train_dataset, val_dataset, coef_norm)
    if mode in {"evaluate", "train_and_evaluate"}:
        if model is None:
            model = load_model(config, device)
        evaluate_model(config, device, model, val_dataset, coef_norm, val_samples, data_dir)


if __name__ == "__main__":
    main()
