import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import torch



def plot_pointcloud_single(
    pos,
    color=None,
    title=None,
    alpha=0.5,
    num_points=10000,
    figsize=(6, 6),
):
    perm = torch.randperm(len(pos), generator=torch.Generator().manual_seed(0))[:num_points]
    plt.close()
    plt.clf()
    view_rotation = [20, 125, 0]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = pos[perm].unbind(-1)
    if color is None:
        scatter = ax.scatter(x, y, z, s=3, c="k", alpha=alpha)
    else:
        scatter = ax.scatter(x, y, z, s=3, c=color[perm], cmap="coolwarm", alpha=alpha)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    plt.axis("equal")
    ax.view_init(*view_rotation)
    if title is not None:
        ax.set_title(title)
    if color is not None:
        plt.colorbar(scatter, orientation="horizontal")
    plt.show()

def plot_pointcloud_double(
    pos,
    color,
    title=None,
    alpha=0.5,
    num_points=10000,
    figsize=(18, 6),
    delta_clamp=None,
    save_path=None
):
    perm = torch.randperm(len(pos[0]), generator=torch.Generator().manual_seed(0))[:num_points]
    plt.close()
    plt.clf()
    view_rotation = [20, 125, 0]
    fig = plt.figure(figsize=figsize)
    axs = []
    vmin = None
    vmax = None
    delta = color[1] - color[0]
    if delta_clamp is not None:
        delta = delta.clamp(*delta_clamp)
    scatters = []

    # ---- 用 GridSpec 定义子图布局 ----
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 0.05], hspace=0.05)
    axs = []
    axs.append(fig.add_subplot(gs[0,0], projection='3d'))
    axs.append(fig.add_subplot(gs[0,1], projection='3d'))
    axs.append(fig.add_subplot(gs[0,2], projection='3d'))

    sub_tags = ["(a)", "(b)", "(c)"]
    caption_text = ""

    for i in range(len(pos) + 1):
        is_delta = i == 2

        if is_delta:
            x, y, z = pos[0][perm].unbind(-1)
        else:
            x, y, z = pos[i][perm].unbind(-1)

        if vmin is None:
            vmin = color[i].min()
            vmax = color[i].max()

        scatter = axs[i].scatter(
            x, y, z, s=3,
            c=delta[perm] if is_delta else color[i][perm],
            cmap="coolwarm",
            vmin=None if is_delta else vmin,
            vmax=None if is_delta else vmax,
            alpha=alpha,
        )
        scatters.append(scatter)
#        axs[i].set_xlabel("X Axis", labelpad=10) # 增加 X 轴标签填充
#        axs[i].set_ylabel("Y Axis", labelpad=10) # 增加 Y 轴标签填充
#        axs[i].set_zlabel("Z Axis", labelpad=10) # 增加 Z 轴标签填充
        axs[i].view_init(*view_rotation)

        if title is not None:
            if is_delta:
                caption_text = f"{sub_tags[i]} delta"
            else:
                caption_text = f"{sub_tags[i]} {title[i]}"

        # 使用 text2D 将文字固定在 Axes 坐标系中
        # x=0.5 表示水平居中
        # y=-0.1 表示放在轴底端向下偏移的位置（根据需要可调整为 -0.15 或 -0.2）
        axs[i].text2D(0.5, 0.1, caption_text,
                      transform=axs[i].transAxes,
                      horizontalalignment='center',
                      verticalalignment='top',
                      fontsize=15,
                      fontname="Times New Roman")


        # title setting
        axs[i].title.set_fontsize(12)
        axs[i].title.set_fontname("Times New Roman")

        # axis label setting
        axs[i].xaxis.label.set_fontsize(12)
        axs[i].xaxis.label.set_fontname("Times New Roman")
        axs[i].yaxis.label.set_fontsize(12)
        axs[i].yaxis.label.set_fontname("Times New Roman")
        axs[i].zaxis.label.set_fontsize(12)
        axs[i].zaxis.label.set_fontname("Times New Roman")

        # tick label setting
        for tick in axs[i].get_xticklabels():
            tick.set_fontsize(12)
            tick.set_fontname("Times New Roman")
        for tick in axs[i].get_yticklabels():
            tick.set_fontsize(12)
            tick.set_fontname("Times New Roman")
        for tick in axs[i].get_zticklabels():
            tick.set_fontsize(12)
            tick.set_fontname("Times New Roman")

        # tick value setting
        axs[i].xaxis.set_major_locator(plt.MaxNLocator(4))
        axs[i].yaxis.set_major_locator(plt.MaxNLocator(4))
        axs[i].zaxis.set_major_locator(plt.MaxNLocator(4))

        # tick line setting
        axs[i].tick_params(width=0.5, pad=2)

        # Set box aspect based one actual data
        data_x_range = x.max() - x.min()
        data_y_range = y.max() - y.min()
        data_z_range = z.max() - z.min()
        axs[i].set_box_aspect((float(data_x_range), float(data_y_range), float(data_z_range)))

    # ---- 统一 colorbar 尺寸 ----

    cbar_ax01 = fig.add_subplot(gs[1, :2])   # 占据前两列的下方空间
    cbar_ax2 = fig.add_subplot(gs[1, 2])   # 占据前两列的下方空间
    fig.colorbar(scatters[0], cax=cbar_ax01, orientation='horizontal')
    fig.colorbar(scatters[2], cax=cbar_ax2, orientation='horizontal')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Figure saved to {save_path}")

    plt.show()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    plot_pointcloud_double(
        pos=[torch.randn(100, 3), torch.randn(100, 3)],
        color=[torch.randn(100), torch.randn(100)],
        title=["target", "prediction"],
    )
