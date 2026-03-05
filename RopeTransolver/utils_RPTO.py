import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.tri import Triangulation
import matplotlib as mpl

def plot_car_results_pointNet(
    x,
    y,
    y_hat,
    save_path,
    figsize=(8, 6),
    point_size=1,
    colorbar_pad=-0.2,       # colorbar与图像下方的距离
    colorbar_fontsize=12,
):
    """
    论文风格点云可视化：
    - 完全隐藏3D坐标轴
    - colorbar 横向紧贴图像下方，长度与图像宽度对齐
    - 点云自动铺满图像区域
    """

    x = np.asarray(x)
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    N = x.shape[1]

    coords = x.reshape(N, 3)
    y_true = y.reshape(N)
    y_pred = y_hat.reshape(N)
    abs_err = np.abs(y_true - y_pred)

    # 自动判断车长/车宽/车高
    ranges = coords.max(axis=0) - coords.min(axis=0)
    sorted_axes = np.argsort(ranges)[::-1]
    long_axis, mid_axis, short_axis = sorted_axes

    X = coords[:, long_axis]
    Y = coords[:, mid_axis]
    Z = coords[:, short_axis]

    x_range, y_range, z_range = X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()
    elev, azim = 25, -135

    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())

    def draw(values, filename, vmin=None, vmax=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点云
        sc = ax.scatter(X, Y, Z, c=values, s=point_size, vmin=vmin, vmax=vmax)

        # 隐藏坐标轴
        ax.set_axis_off()

        # 保持长宽高比例
        eps = 1e-8
        ax.set_box_aspect((max(x_range, eps), max(y_range, eps), max(z_range, eps)))
        ax.view_init(elev=elev, azim=azim)

        # 设置点云显示范围，让点云尽量填满图片
        margin = 0.02
        ax.set_xlim(X.min() - x_range*margin, X.max() + x_range*margin)
        ax.set_ylim(Y.min() - y_range*margin, Y.max() + y_range*margin)
        ax.set_zlim(Z.min() - z_range*margin, Z.max() + z_range*margin)

        # colorbar 紧贴图像下方，长度与图像宽度对齐
        cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=colorbar_pad, fraction=0.05)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)

    draw(y_true, f"{save_path}_true.png", vmin, vmax)
    draw(y_pred, f"{save_path}_pred.png", vmin, vmax)
    draw(abs_err, f"{save_path}_abs_error.png")

    print("Saved paper-style point cloud images with tight colorbar.")


def plot_car_vel_pointNet(
    x,
    y,
    y_hat,
    save_path,
    figsize=(8, 6),
    point_size=1,
    colorbar_pad=-0.2,
    colorbar_fontsize=12,
    elev=25,
    azim=-135,
    focus_method="percentile",   # "percentile" | "voxel" | "surface" | None
    lower_pct=1.0,               # for percentile method: lower percentile (0-100)
    upper_pct=99.0,              # for percentile method: upper percentile (0-100)
    voxel_grid=50,               # for voxel method: grid size per axis
    focus_radius_scale=0.5,      # for voxel/surface: radius = scale * max(range)
    surface_idx=None,            # for surface method: indices (1D array) of surface points in coords
    downsample_farfield=None,    # if int N: draw up to N farfield points (random sample) as background (optional)
):
    """
    改进版点云可视化，支持自动聚焦车辆区域以避免外流点把车画得很小。

    参数说明（新增关键）：
    - focus_method:
        - "percentile": 按坐标的上下 percentiles 裁剪（默认），去除极端远场点
        - "voxel": 基于体素计数，找到最密集体素并以其为中心裁剪半径区域
        - "surface": 如果你能给出车表点索引 surface_idx，会以表面质心为中心裁剪
        - None: 不裁剪（原行为）
    - lower_pct / upper_pct: percentile 裁剪阈值（默认 1% - 99%）
    - voxel_grid: 体素网格分辨率（越大越精细但越耗内存）
    - focus_radius_scale: 裁剪半径相对于坐标最大范围的缩放因子（0-1）
    - downsample_farfield: 若指定整数 N，会在绘主视图前把被裁掉的远场点随机抽样 N 个作为背景一起绘（可选）
    - surface_idx: 若使用 "surface" 方法，需要传入对应的点索引（array-like）
    """

    x = np.asarray(x)
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)

    # 规范到 (N, 3) / (N,) 形式
    # 支持两种输入：(1,N,3) 或 (N,3)
    if x.ndim == 3 and x.shape[0] == 1:
        coords = x.reshape(-1, 3)
    elif x.ndim == 2 and x.shape[1] == 3:
        coords = x
    else:
        raise ValueError("x must be shape (1,N,3) or (N,3)")

    # y / y_hat -> (N,)
    y_true = y.reshape(-1)
    y_pred = y_hat.reshape(-1)
    abs_err = np.abs(y_true - y_pred)

    # 先计算原始范围（用于后续尺度与裁剪）
    full_ranges = coords.max(axis=0) - coords.min(axis=0)
    global_max_range = max(full_ranges.max(), 1e-8)

    # ----------------- 选择裁剪策略，生成 mask_keep -----------------
    Npts = coords.shape[0]
    mask_keep = np.ones(Npts, dtype=bool)

    if focus_method is None:
        mask_keep = np.ones(Npts, dtype=bool)
    elif focus_method == "percentile":
        # 按坐标 axis 的 percentile 裁剪，去掉极端远场点
        low = np.percentile(coords, lower_pct, axis=0)
        high = np.percentile(coords, upper_pct, axis=0)
        mask_keep = np.ones(Npts, dtype=bool)
        for d in range(3):
            mask_keep &= (coords[:, d] >= low[d]) & (coords[:, d] <= high[d])
    elif focus_method == "voxel":
        # 把点放入体素网格，找到点数最多的体素，以体素中心为 focus_center
        # 然后保留以该中心为球心、radius = focus_radius_scale * global_max_range 的点
        # 这个方法对自动找“车辆附近最密集点簇”常有效
        # 为避免内存问题，使用 np.histogramdd（不显式生成全 NxN matrix）
        grids = [voxel_grid, voxel_grid, voxel_grid]
        H, edges = np.histogramdd(coords, bins=grids)
        # 找到最大点数体素索引
        max_idx = np.unravel_index(np.argmax(H), H.shape)
        # 计算体素中心
        centers = []
        for dim in range(3):
            ed = edges[dim]
            # center of bin max_idx[dim] is (edges[i] + edges[i+1]) / 2
            i = max_idx[dim]
            center = 0.5 * (ed[i] + ed[i+1])
            centers.append(center)
        focus_center = np.array(centers)
        radius = focus_radius_scale * global_max_range
        dists = np.linalg.norm(coords - focus_center[None, :], axis=1)
        mask_keep = dists <= radius
    elif focus_method == "surface":
        # surface_idx 应该给出车辆表面点的索引（一维数组）
        if surface_idx is None:
            raise ValueError("surface_idx must be provided for focus_method='surface'")
        surface_coords = coords[np.asarray(surface_idx)]
        center = surface_coords.mean(axis=0)
        # 可以用表面最大半径作为基准
        surf_ranges = surface_coords.max(axis=0) - surface_coords.min(axis=0)
        radius = focus_radius_scale * max(surf_ranges.max(), 1e-8)
        dists = np.linalg.norm(coords - center[None, :], axis=1)
        mask_keep = dists <= radius
    else:
        raise ValueError(f"unknown focus_method: {focus_method}")

    # 如果裁剪后点太少（比如阈值太严），则回退到更宽松的 percentile 5-95
    if mask_keep.sum() < max(100, int(0.01 * Npts)):
        # 回退策略
        low = np.percentile(coords, 5.0, axis=0)
        high = np.percentile(coords, 95.0, axis=0)
        mask_keep = np.ones(Npts, dtype=bool)
        for d in range(3):
            mask_keep &= (coords[:, d] >= low[d]) & (coords[:, d] <= high[d])
        # 仍然太少就不裁剪
        if mask_keep.sum() < 50:
            mask_keep = np.ones(Npts, dtype=bool)

    # 另：准备 farfield 被裁掉的点（可选下采样并画为背景）
    mask_far = ~mask_keep
    if downsample_farfield is not None and mask_far.sum() > 0:
        k = int(downsample_farfield)
        idx_far = np.where(mask_far)[0]
        if idx_far.size > k:
            chosen = np.random.choice(idx_far, size=k, replace=False)
        else:
            chosen = idx_far
        mask_far_plot = np.zeros(Npts, dtype=bool)
        mask_far_plot[chosen] = True
    else:
        mask_far_plot = np.zeros(Npts, dtype=bool)

    # 选取要绘制的点
    coords_focus = coords[mask_keep]
    y_true_focus = y_true[mask_keep]
    y_pred_focus = y_pred[mask_keep]
    abs_err_focus = abs_err[mask_keep]

    coords_far = coords[mask_far_plot]
    y_true_far = y_true[mask_far_plot]
    y_pred_far = y_pred[mask_far_plot]
    abs_err_far = abs_err[mask_far_plot]

    # 重新定义 X,Y,Z：保持与原函数相同的“自动长/中/短轴”映射，以保证视角一致
    ranges = coords_focus.max(axis=0) - coords_focus.min(axis=0)
    sorted_axes = np.argsort(ranges)[::-1]
    long_axis, mid_axis, short_axis = sorted_axes

    X = coords_focus[:, long_axis]
    Y = coords_focus[:, mid_axis]
    Z = coords_focus[:, short_axis]

    x_range, y_range, z_range = X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()
    eps = 1e-8

    vmin = min(y_true_focus.min(), y_pred_focus.min())
    vmax = max(y_true_focus.max(), y_pred_focus.max())

    def draw(values_focus, filename, vmin=None, vmax=None, draw_farfield=False, values_far=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 先画 farfield（若有），用极小点和 alpha 做背景参考
        if draw_farfield and coords_far.size > 0:
            Xf = coords_far[:, long_axis]
            Yf = coords_far[:, mid_axis]
            Zf = coords_far[:, short_axis]
            scf = ax.scatter(Xf, Yf, Zf, c=values_far, s=max(0.1, point_size*0.2), alpha=0.25, vmin=vmin, vmax=vmax)

        sc = ax.scatter(X, Y, Z, c=values_focus, s=point_size, vmin=vmin, vmax=vmax)

        ax.set_axis_off()
        ax.set_box_aspect((max(x_range, eps), max(y_range, eps), max(z_range, eps)))
        ax.view_init(elev=elev, azim=azim)

        margin = 0.02
        ax.set_xlim(X.min() - x_range*margin, X.max() + x_range*margin)
        ax.set_ylim(Y.min() - y_range*margin, Y.max() + y_range*margin)
        ax.set_zlim(Z.min() - z_range*margin, Z.max() + z_range*margin)

        # colorbar 横向紧贴下方，长度与图片宽度对齐
        cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=colorbar_pad, fraction=0.05)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)

    # 绘三张图：true / pred / abs error
    draw(y_true_focus, f"{save_path}_true.png", vmin, vmax, draw_farfield=(mask_far_plot.sum()>0), values_far=y_true_far)
    draw(y_pred_focus, f"{save_path}_pred.png", vmin, vmax, draw_farfield=(mask_far_plot.sum()>0), values_far=y_pred_far)
    draw(abs_err_focus, f"{save_path}_abs_error.png", None, None, draw_farfield=(mask_far_plot.sum()>0), values_far=abs_err_far)

    print("Saved focused point cloud images (car-centered) with tight colorbar.")