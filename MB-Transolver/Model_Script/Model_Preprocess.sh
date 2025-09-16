python run_pipeline.py \
    --stages preprocess \
    --exp_name "Test" \
    --Wdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/E_S_WW_WM" \
    --Wcache_dir "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/Cache_data-E_S_WW_WM" \
    --Pdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/E_S_WW_WM" \
    --Pcache_dir "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/Cache_data-E_S_WW_WM" \
    --Cdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/E_S_WW_WM" \
    --Ccache_dir "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/Cache_data-E_S_WW_WM" \
    --Vdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/E_S_WW_WM" \
    --Vcache_dir "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/Cache_data-E_S_WW_WM" \
    --subset_dir "/work/mae-zhangbj/ML_Turbulent/Current_Work/MB-Transolver/train_val_test_splits" \
    --num_points 10000

#######################################################
# Wall Shear Stress
# ~~~~~~~~~~~~~~~~~~
#

# E_S_WW_WM
# --Wdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/E_S_WW_WM"
# --Wcache_dir "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/Cache_data-E_S_WW_WM"

