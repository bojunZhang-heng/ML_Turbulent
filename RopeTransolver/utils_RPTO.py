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
