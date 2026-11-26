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
# DrivAerStar
#nohup python main.py --config config_RETO_plus.yml > ${LOG_DIR}/RETO_plus_DrivAerStar_${TIMESTAMP}.log 2>&1 &
#nohup python main.py --config config_CANO.yml > ${LOG_DIR}/CANO_DrivAerStar_${TIMESTAMP}.log 2>&1 &
#nohup python main.py --config config_Transolver.yml > ${LOG_DIR}/Transolver_DrivAerStar_${TIMESTAMP}.log 2>&1 &

#nohup python main.py --config config_RETO_plus_v2.yml > ${LOG_DIR}/RETO_plus_v2_DrivAerStar_${TIMESTAMP}.log 2>&1 &
#nohup python main.py --config config_RETO.yml > ${LOG_DIR}/RETO_DrivAerStar_${TIMESTAMP}.log 2>&1 &

# postprocess
nohup python ./evaluate_models.py \
  --runs-file runs_checkpoint.yml \
  --output-dir postprocess_results/pressure_900 \
  > ${LOG_DIR}/post-Multi_model_DrivAerStar_${TIMESTAMP}.log 2>&1 &



# 获取训练脚本 PID
PID=$!

# 记录 PID + 时间
PID_LOG=${LOG_DIR}/PID_${PID}_${TIMESTAMP}.log
echo "$(date '+%Y-%m-%d %H:%M:%S') | Training PID: $PID" > $PID_LOG

# 同时在终端显示
cat $PID_LOG

