import numpy as np
from utils_RPTO import plot_car_DrivAerML_velocity 
import os

size=np.load("/Users/zhangbojun/ML_Turbulent/RopeTransolver/data_DrivAerML/pressure/x.npy")
x = np.load("/Users/zhangbojun/ML_Turbulent/RopeTransolver/data_DrivAerML/pressure/x.npy")
y = np.load("/Users/zhangbojun/ML_Turbulent/RopeTransolver/data_DrivAerML/pressure/y.npy")
y_hat = np.load("/Users/zhangbojun/ML_Turbulent/RopeTransolver/data_DrivAerML/pressure/y_hat.npy")

size = size[0]
x = x[0]       # (10000,3)
y_hat = y_hat[0]    # (10000,)
y = np.linalg.norm(y, axis=1, keepdims=True)
y_hat = np.linalg.norm(y_hat, axis=1, keepdims=True)

print(x.shape)
print(y.shape)
print(y_hat.shape)

figure_path = "/Users/zhangbojun/ML_Turbulent/RopeTransolver/fig/pressure/"
plot_car_DrivAerML_velocity(
    size,
    x, y, y_hat,
    save_path=figure_path,
    figsize=(12, 8),
    colorbar_fontsize=24,
)
# 24
# 26