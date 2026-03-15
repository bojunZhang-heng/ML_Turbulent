#!/bin/bash
# cuda:0 for DrivAerNet++ spressure swss Ropesolver
# cuda:1 for DrivAerNet++ spressure swss Transolver
# cuda:2 for DrivAerNet   spressure swss Ropesolver
# cuda:3 for DrivAerNet   spressure swss Transolver

# One GPU test one case
# cuda:2 for DrivAerNet++ spressure swss Ropesolver
# cuda:5 for DrivAerNet spressure  Ropesolver

# This time
# cuda:5
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# 确保 logs 目录存在
LOG_DIR=./logs
mkdir -p $LOG_DIR

# 用时间命名日志
TIMESTAMP=$(date "+%Y_%m%d_%H%M")

# 启动训练（后台）
# DrivAerML
#nohup python train_DrivAerML.py config_DrivAerML_velocity.yml > ${LOG_DIR}/DrivAerML_velocity_${TIMESTAMP}.log 2>&1 &
nohup python train_DrivAerML.py config_DrivAerML_spressure.yml > ${LOG_DIR}/DrivAerML_spressure_${TIMESTAMP}.log 2>&1 &
#nohup python train_DrivAerML.py config_DrivAerML_swss.yml > ${LOG_DIR}/DrivAerML_swss_${TIMESTAMP}.log 2>&1 &
#nohup python test_model.py config_DrivAerML_velocity.yml > ${LOG_DIR}/DrivAerML_test_${TIMESTAMP}.log 2>&1 &

# DrivAerNet
#nohup python train_DrivAerNet.py config_DrivAerNet_spressure.yml > ${LOG_DIR}/DrivAerNet_spressure_${TIMESTAMP}.log 2>&1 &
#nohup python train_DrivAerNet.py config_DrivAerNet_swss.yml > ${LOG_DIR}/DrivAerNet_swss_${TIMESTAMP}.log 2>&1 &

# DrivAerNet++
#nohup python train_DrivAerNet++.py config_DrivAerNet++_WW_spressure.yml > ${LOG_DIR}/DrivAerNet++_WW_spressure_${TIMESTAMP}.log 2>&1 &
#nohup python train_DrivAerNet++.py config_DrivAerNet++_WWC_spressure.yml > ${LOG_DIR}/DrivAerNet++_WWC_spressure_${TIMESTAMP}.log 2>&1 &
#nohup python train_DrivAerNet++.py config_DrivAerNet++_WW_swss.yml > ${LOG_DIR}/DrivAerNet++_WW_swss_${TIMESTAMP}.log 2>&1 &
#nohup python train_DrivAerNet++.py config_DrivAerNet++_WWC_swss.yml > ${LOG_DIR}/DrivAerNet++_WWC_swss_${TIMESTAMP}.log 2>&1 &

# AhmedBody
#nohup python train_AhmedBody.py config_AhmedBody.yml > ${LOG_DIR}/AhmedBody_${TIMESTAMP}.log 2>&1 &

# ShapeNet
#nohup python train_ShapeNet.py config_ShapeNet.yml > ${LOG_DIR}/ShapeNet_${TIMESTAMP}.log 2>&1 &

# 获取训练脚本 PID
PID=$!

# 记录 PID + 时间
PID_LOG=${LOG_DIR}/PID_${PID}_${TIMESTAMP}.log
echo "$(date '+%Y-%m-%d %H:%M:%S') | Training PID: $PID" > $PID_LOG

# 同时在终端显示
cat $PID_LOG

