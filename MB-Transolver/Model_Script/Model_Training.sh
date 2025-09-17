python ./run_pipeline.py \
    --stages train \
    --exp_name "Test" \
    --model "Transolver_Irregular_Mesh" \
    --num_points 10000 \
    --Wdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/E_S_WW_WM" \
    --Wcache_dir "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/Cache_data-E_S_WW_WM" \
    --Pdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/E_S_WW_WM" \
    --Pcache_dir "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/Cache_data-E_S_WW_WM" \
    --Cdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/CAD/E_S_WW_WM" \
    --Ccache_dir "/work/mae-zhangbj/DrivAriNet_dataset/CAD/Cache_data-E_S_WW_WM" \
    --Vdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/E_S_WW_WM" \
    --Vcache_dir "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/Cache_data-E_S_WW_WM" \
    --subset_dir "/work/mae-zhangbj/ML_Turbulent/Current_Work/MB-Transolver/train_val_test_splits" \
    --n_hidden 128 \
    --n_heads 8 \
    --n_layers 8 \
    --lr 0.001 \
    --max_grad_norm 0.1 \
    --slice_num 64 \
    --unified_pos 1 \
    --ref 8 \
    --downsample 5 \
    --num_workers 1 \
    --batch_size 6 \
    --epochs 500 \
    --test_only 0 \
    --gpus "0" 


# ------------------small_samples
    #--dataset_path "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/small_samples/Pressure_VTK" \
    #--subset_dir "/work/mae-zhangbj/ML_Turbulent/Current_Work/Pressure_Field/train_val_test_splits" \
    #--cache_dir "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/small_samples/Cache_data" \

# ------------------E_S_WWC_WM
    #--dataset_path "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/E_S_WWC_WM" \
    #--subset_dir "/work/mae-zhangbj/ML_Turbulent/Current_Work/Pressure_Field/train_val_test_splits" \
    #--cache_dir "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/E_S_WWC_WM/Cache_data" \

# ------------------E_S_WW_WM
    #--dataset_path "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/E_S_WW_WM" \
    #--subset_dir "/work/mae-zhangbj/ML_Turbulent/Current_Work/Pressure_Field/train_val_test_splits" \
    #--cache_dir "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/E_S_WW_WM/Cache_data" \

# ------------------N_S_WWS_WM
    #--dataset_path "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/N_S_WWS_WM" \
    #--subset_dir "/work/mae-zhangbj/ML_Turbulent/Current_Work/Pressure_Field/train_val_test_splits" \
    #--cache_dir "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/N_S_WWS_WM/Cache_data" \
