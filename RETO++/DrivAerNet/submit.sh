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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=4

# 确保 logs 目录存在
LOG_DIR=./logs
mkdir -p $LOG_DIR

# 用时间命名日志
TIMESTAMP=$(date "+%Y_%m%d_%H%M")



# 启动训练（后台）
# DrivAerNet pressure
nohup python main.py --config config_CANO_pressure.yml > ${LOG_DIR}/CANO_DrivAerNet_pressure_${TIMESTAMP}.log 2>&1 &
nohup python main.py --config config_Transolver_pressure.yml > ${LOG_DIR}/Transolver_DrivAerNet_pressure_${TIMESTAMP}.log 2>&1 &
nohup python main.py --config config_RETO_plus_v2_pressure.yml > ${LOG_DIR}/RETO_plus_v2_DrivAerNet_pressure_${TIMESTAMP}.log 2>&1 &
nohup python main.py --config config_RETO_plus_pressure.yml > ${LOG_DIR}/RETO_plus_DrivAerNet_pressure_${TIMESTAMP}.log 2>&1 &
nohup python main.py --config config_LinearNO_pressure.yml > ${LOG_DIR}/LinearNO_DrivAerNet_pressure_${TIMESTAMP}.log 2>&1 &
nohup python main.py --config config_SATO_pressure.yml > ${LOG_DIR}/SATO_DrivAerNet_pressure_${TIMESTAMP}.log 2>&1 &
#nohup python main.py --config config_RETO.yml > ${LOG_DIR}/RETO_DrivAerNet_pressure_${TIMESTAMP}.log 2>&1 &


# 获取训练脚本 PID
PID=$!

# 记录 PID + 时间
PID_LOG=${LOG_DIR}/PID_${PID}_${TIMESTAMP}.log
echo "$(date '+%Y-%m-%d %H:%M:%S') | Training PID: $PID" > $PID_LOG

# 同时在终端显示
cat $PID_LOG

