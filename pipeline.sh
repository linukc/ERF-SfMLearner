# python3.7 data/prepare_train_data.py "/media/serlini/data3/Datasets/kitti_odometry/dataset" \
#                     --dataset-format 'kitti_odometry' --dump-root "/media/serlini/data3/Datasets/kitti_odometry/dataset_formatted" \
#                     --width 416 --height 128 --num-threads 4 --static-frames "data/static_frames.txt" \
#                     --with-pose # 23 min

# общее количество файлов примерно 40к

python3.7 train.py "/media/serlini/data3/Datasets/kitti_odometry/dataset_formatted" -b4 -m0 -s2.0 --epoch-size 1000 --sequence-length 5 --log-output --with-pose