python3 data/prepare_train_data.py "/mnt/hdd4/Datasets/Kitti/dataset/" \
                     --dataset-format 'kitti_odometry' --dump-root "/mnt/hdd4/Datasets/Kitti/dataset/formatted_310_94" \
                     --width 310 --height 94 --num-threads 4 --static-frames "data/static_frames.txt" \
                     --with-pose # 23 min

python3 data/prepare_train_data.py "/mnt/hdd4/Datasets/Kitti/dataset/" \
                     --dataset-format 'kitti_odometry' --dump-root "/mnt/hdd4/Datasets/Kitti/dataset/formatted_248_75" \
                     --width 248 --height 75 --num-threads 4 --static-frames "data/static_frames.txt" \
                     --with-pose

python3 data/prepare_train_data.py "/mnt/hdd4/Datasets/Kitti/dataset/" \
                     --dataset-format 'kitti_odometry' --dump-root "/mnt/hdd4/Datasets/Kitti/dataset/formatted_416_128" \
                     --width 416 --height 128 --num-threads 4 --static-frames "data/static_frames.txt" \
                     --with-pose

# общее количество файлов примерно 40к

#python3.7 train.py "/media/serlini/data3/Datasets/kitti_odometry/dataset_formatted" -b4 -m0 -s2.0 --epoch-size 1000 --sequence-length 5 --log-output --with-pose
