python ./run_pipeline.py \
    --stages train \
    --exp_name "Test" \
    --num_points 10000 \
    --Wdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/E_S_WW_WM" \
    --Wcache_dir "/work/mae-zhangbj/DrivAriNet_dataset/WallShearStress/Cache_data-E_S_WW_WM" \
    --Pdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/E_S_WW_WM" \
    --Pcache_dir "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/Cache_data-E_S_WW_WM" \
    --Cdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/CAD/E_S_WW_WM" \
    --Ccache_dir "/work/mae-zhangbj/DrivAriNet_dataset/CAD/Cache_data-E_S_WW_WM" \
    --Vdataset_path "/work/mae-zhangbj/DrivAriNet_dataset/Volumetric_Field/E_S_WW_WM" \
    --Vcache_dir "/work/mae-zhangbj/DrivAriNet_dataset/Volumetric_Field/Cache_data-E_S_WW_WM" \
    --subset_dir "/work/mae-zhangbj/ML_Turbulent/Current_work/MB-Transolver/train_val_test_splits" \
    --lr 0.001 \
    --max_grad_norm 0.1 \
    --slice_num 64 \
    --ref 8 \
    --downsample 5 \
    --num_workers 1 \
    --batch_size 6 \
    --epochs 500 \
    --test_only 0 \
    --gpus "0" \
    --n_dim 3  \
    --input_dim 3 \
    --output_dim_surface 4 \
    --output_dim_volume  7 \
    --geometry_depth 1 \
    --n_heads 4 \
    --blocks "s" \
    --num_volume_blocks 6 \
    --num_surface_blocks 6 \
    --n_input 128 \
    --n_hidden 128 \
    --n_layers 8 \
    --res "True" \
    --dim_head 64 

#    --blocks "pscscs" \
#    --n_dim 3 \                       number of coordinates (typicaly 3 for 3D geometries)
#    --input_dim 3 \                   we only use coordinates as inputs
#    --output_dim_surface 4 \          surface pressure (1D) and wallshearstress (3D)
#    --output_dim_volume  7 \          volume pressure (1D), volume velocity (3D) and volume vorticity (3D)
#    --geometry_depth 1 \              1 transformer block to encode the geometry
#    --num_heads 4 \                   number of attention heads in a ViT-tiny
#    --num_volume_blocks 6 \           6 modality-specific self-attention blocks
#    --num_surface_blocks 6 \          6 modality-specific self-attention blocks

# "p": weight-shared cross-attention block to the geometry branch outputs
# "s" weight-shared split attention block within surface/volume
# "c" weight-shared cross-attention block between surface/volume

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
