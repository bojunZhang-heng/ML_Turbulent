#!/bin/bash

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=5

# 确保 logs 目录存在
LOG_DIR=./logs
mkdir -p $LOG_DIR

# 用时间命名日志
TIMESTAMP=$(date "+%Y_%m%d_%H%M")

# 启动训练（后台）
#nohup python train.py config_train_s_pressure.yml > ${LOG_DIR}/DrivAerML_pressure_${TIMESTAMP}.log 2>&1 &
#nohup python train.py config_train_velocity.yml > ${LOG_DIR}/DrivAerML_velocity_${TIMESTAMP}.log 2>&1 &
nohup python train_drivaernet.py config_train_DrivAerNet_pressure.yml > ${LOG_DIR}/DrivAerNet_pressure_${TIMESTAMP}.log 2>&1 &
#nohup python train_drivaernet.py config_train_DrivAerNet_wss.yml > ${LOG_DIR}/DrivAerNet_wss_${TIMESTAMP}.log 2>&1 &


# 获取训练脚本 PID
PID=$!

# 记录 PID + 时间
PID_LOG=${LOG_DIR}/PID_${PID}_${TIMESTAMP}.log
echo "$(date '+%Y-%m-%d %H:%M:%S') | Training PID: $PID" > $PID_LOG

# 同时在终端显示
cat $PID_LOG

