#!/bin/bash
#SBATCH -J RT_DrivAerNet            # 作业名
#SBATCH -p dzagnormal            # 分区（你当前的分区）
#SBATCH --nodes=1                # 节点数
#SBATCH --ntasks-per-node=1      # 每个节点任务数（通常为1）
#SBATCH --cpus-per-task=4        # 每个任务的 CPU 核心数
#SBATCH --gres=gpu:1             # 请求 GPU 数量（1 个 GPU）
#SBATCH -o logs/%j.out                # 标准输出文件名（%j = jobid）
#SBATCH -e logs/%j.err                # 标准错误文件

source ~/miniconda3/etc/profile.d/conda.sh 
conda activate Rope

# 指定使用 gpu16 上的 GPU 0
#export CUDA_VISIBLE_DEVICES=0

# 打印一些调试信息（可选）
echo "JobID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Gres (requested): $SLURM_JOB_GPUS"
echo "CPUs per task: $SLURM_CPUS_ON_NODE"

# 查看分配到的 GPU（在作业内执行）
nvidia-smi

# 推荐用 srun 来启动程序（更符合 SLURM 调度）
python train.py config_train_s_pressure.yml
