import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter

def plot_car_ShapeNet_pressure(
    size,
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
    # ======================
    # 全局字体：Times New Roman
    # ======================
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False
    
    size = np.asarray(size)
    x = np.asarray(x)
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)

    # 规范到 (N, 3) / (N,) 形式
    # 支持两种输入：(1,N,3) 或 (N,3)
    if x.ndim == 3 and x.shape[0] == 1:
        coords = size.reshape(-1, 3)
    elif x.ndim == 2 and x.shape[1] == 3:
        coords = size
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


def plot_car_DrivAerML_velocity(
    size,      # 速度场空间位置，用于裁剪和固定坐标轴比例 (N_s,3) 或 (1,N_s,3)
    x,         # y/y_hat 的空间位置 (1,N,3) 或 (N,3)
    y,         # 标量值 (1,N,1) 或 (N,1)
    y_hat,     # 标量值 (1,N,1) 或 (N,1)
    save_path="fig",
    figsize=(8,6),
    point_size=1,
    colorbar_pad=-0.2,
    colorbar_fontsize=12,
    elev=25,
    azim=-135,
    focus_method="percentile",  # "percentile" | "voxel" | "surface" | None
    lower_pct=1.0,
    upper_pct=99.0,
    voxel_grid=50,
    focus_radius_scale=0.5,
    surface_idx=None,
    downsample_farfield=None,
):
    # ======================
    # 全局字体：Times New Roman
    # ======================
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.unicode_minus"] = False

    # --- 处理输入维度 ---
    if size.ndim == 3 and size.shape[0]==1:
        size = size[0]
    if x.ndim == 3 and x.shape[0]==1:
        coords = x[0]
    else:
        coords = x
    if y.ndim == 3 and y.shape[0]==1:
        y_true = y[0].reshape(-1)
    else:
        y_true = y.reshape(-1)
    if y_hat.ndim == 3 and y_hat.shape[0]==1:
        y_pred = y_hat[0].reshape(-1)
    else:
        y_pred = y_hat.reshape(-1)
    abs_err = np.abs(y_true - y_pred)

    # --- focus_method 裁剪 ---
    Npts = size.shape[0]
    mask_keep = np.ones(Npts, dtype=bool)
    if focus_method == "percentile":
        low = np.percentile(size, lower_pct, axis=0)
        high = np.percentile(size, upper_pct, axis=0)
        for d in range(3):
            mask_keep &= (size[:,d]>=low[d]) & (size[:,d]<=high[d])
    elif focus_method == "voxel":
        H, edges = np.histogramdd(size, bins=[voxel_grid]*3)
        max_idx = np.unravel_index(np.argmax(H), H.shape)
        centers = [0.5*(edges[d][max_idx[d]]+edges[d][max_idx[d]+1]) for d in range(3)]
        focus_center = np.array(centers)
        radius = focus_radius_scale * np.max(size.max(axis=0)-size.min(axis=0))
        mask_keep = np.linalg.norm(size - focus_center[None,:], axis=1)<=radius
    elif focus_method == "surface":
        if surface_idx is None:
            raise ValueError("surface_idx must be provided for surface focus")
        surface_coords = size[surface_idx]
        center = surface_coords.mean(axis=0)
        radius = focus_radius_scale * np.max(surface_coords.max(axis=0)-surface_coords.min(axis=0))
        mask_keep = np.linalg.norm(size - center[None,:], axis=1)<=radius
    elif focus_method is None:
        mask_keep = np.ones(Npts, dtype=bool)
    else:
        raise ValueError(f"Unknown focus_method: {focus_method}")

    # 用裁剪后的速度场范围固定坐标轴
    size_focus = size[mask_keep]
    xyz_range = size_focus.max(axis=0) - size_focus.min(axis=0)
    xyz_min = size_focus.min(axis=0)
    xyz_max = size_focus.max(axis=0)

    # --- 绘图函数 ---
    def draw(values, filename, vmin=None, vmax=None):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2],
                        c=values, s=point_size, vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        ax.set_box_aspect(xyz_range)
        ax.set_xlim(xyz_min[0], xyz_max[0])
        ax.set_ylim(xyz_min[1], xyz_max[1])
        ax.set_zlim(xyz_min[2], xyz_max[2])
        ax.view_init(elev=elev, azim=azim)

        cbar = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=colorbar_pad, fraction=0.05)
        # ===== 手动设置 5 个刻度 =====
        if vmin is not None and vmax is not None:
            ticks = np.linspace(vmin, vmax, 5)
            cbar.set_ticks(ticks)
        else:
            # 对于 error（没有固定范围），用当前数据范围
            ticks = np.linspace(values.min(), values.max(), 5)
            cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)

    vmin = y_true.min()
    vmax = y_true.max()
    draw(y_true, f"{save_path}_true.png", vmin, vmax)
    draw(y_pred, f"{save_path}_pred.png", vmin, vmax)
    draw(abs_err, f"{save_path}_abs_error.png", None, None)
    print("Saved point cloud images with focus-based axes (car proportion preserved).")
