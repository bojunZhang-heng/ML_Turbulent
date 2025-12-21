#!/bin/bash
# cuda:3 for DrivAerNet
# cuda:4 for DrivAerML velocity run_*
# cuda:5 for DrivAerNet

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=4

# 确保 logs 目录存在
LOG_DIR=./logs
mkdir -p $LOG_DIR

# 用时间命名日志
TIMESTAMP=$(date "+%Y_%m%d_%H%M")

# 启动训练（后台）
# DrivAerML
nohup python train_DrivAerML.py config_DrivAerML_spressure.yml > ${LOG_DIR}/DrivAerML_spressure_${TIMESTAMP}.log 2>&1 &
#nohup python train_DrivAerML.py config_DrivAerML_velocity.yml > ${LOG_DIR}/DrivAerML_velocity_${TIMESTAMP}.log 2>&1 &

# DrivAerNet
#nohup python train_DrivAerNet.py config_DrivAerNet_spressure.yml > ${LOG_DIR}/DrivAerNet_spressure_${TIMESTAMP}.log 2>&1 &
#nohup python train_drivaernet.py config_DrivAerNet_swss.yml > ${LOG_DIR}/DrivAerNet_swss_${TIMESTAMP}.log 2>&1 &


# 获取训练脚本 PID
PID=$!

# 记录 PID + 时间
PID_LOG=${LOG_DIR}/PID_${PID}_${TIMESTAMP}.log
echo "$(date '+%Y-%m-%d %H:%M:%S') | Training PID: $PID" > $PID_LOG

# 同时在终端显示
cat $PID_LOG

