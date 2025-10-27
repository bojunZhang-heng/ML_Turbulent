from drivaerml_dataset import DrivAerMLDataset
from abupt_collator import AbuptCollator
from torch.utils.data import DataLoader

def create_data_loaders(root_dir, batch_size, num_sample_frac=0.001, num_workers=1):
    """创建训练、验证和测试数据加载器"""

    # 创建数据集
    train_dataset = DrivAerMLDataset(root=root_dir, split="train", num_sample_frac=num_sample_frac)
    val_dataset = DrivAerMLDataset(root=root_dir, split="val", num_sample_frac=num_sample_frac)
    test_dataset = DrivAerMLDataset(root=root_dir, split="test", num_sample_frac=num_sample_frac)

    # 设置采样参数 - 根据你的需求调整这些参数
    num_geometry_points = 65536  # 几何点数量
    num_surface_anchor_points = 16384  # 表面锚点数量
    num_volume_anchor_points =  16384 # 体积锚点数量
    num_geometry_supernodes = 16384  # 几何超节点数量

    # 创建collator
    train_collator = AbuptCollator(
        num_geometry_points=num_geometry_points,
        num_surface_anchor_points=num_surface_anchor_points,
        num_volume_anchor_points=num_volume_anchor_points,
        num_geometry_supernodes=num_geometry_supernodes,
        dataset=train_dataset,
        use_query_positions=True,  # 根据你的模型需求设置
        seed=42  # 为了可重现性
    )

    # 验证和测试可以使用相同的collator配置
    val_collator = AbuptCollator(
        num_geometry_points=num_geometry_points,
        num_surface_anchor_points=num_surface_anchor_points,
        num_volume_anchor_points=num_volume_anchor_points,
        num_geometry_supernodes=num_geometry_supernodes,
        dataset=val_dataset,
        use_query_positions=True,
        seed=42
    )

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collator,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collator,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_collator,  # 使用val的collator
        pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader
