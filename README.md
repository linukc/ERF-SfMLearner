### ERF-SfMLearner: Extended receptive field for Structure from Motion Learner
This codebase implements the system described in the paper *"Influence of Neural Network Receptive Field on Monocular Depth and Ego-Motion Estimation"* and bases on the [SfMLearner pytorch version](https://github.com/ClementPinard/SfmLearner-Pytorch).

#### Load dataset

For KITTI, first download the dataset using this script provided on the official website, and then run the following command. The --with-depth option will save resized copies of groundtruth to help you setting hyper parameters. The --with-pose will dump the sequence pose in the same format as Odometry dataset (see pose evaluation)

```python
python3 data/prepare_train_data.py /path/to/raw/kitti/dataset/ --dataset-format 'kitti_raw' --dump-root /path/to/resulting/formatted/data/ --width 416 --height 128 --num-threads 4 [--static-frames /path/to/static_frames.txt] [--with-depth] [--with-pose]
```

#### Train network

Once the data are formatted following the above instructions, you should be able to train the model by running the following command

```python
python3 train.py /path/to/the/formatted/data/ -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 3 --log-output [--with-gt]
```
Pay attention to the models folder - there you will find models with extended receptive field.

#### Eval

Depth and pose evaluation scipts are *test_disp.py* and *test_pose.py*.
