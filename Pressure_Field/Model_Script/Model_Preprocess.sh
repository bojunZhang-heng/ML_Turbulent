python run_pipeline.py \
    --stages preprocess \
    --exp_name "Test" \
    --dataset_path "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/small_samples/Pressure_VTK" \
    --subset_dir "/work/mae-zhangbj/ML_Turbulent/Current_Work/Pressure_Field/train_val_test_splits" \
    --cache_dir "/work/mae-zhangbj/DrivAriNet_dataset/Pressure_Field/small_samples/Cache_data" \
    --num_points 10000

