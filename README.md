# LSNet: LONG SHORT TEMPORAL NETWORK FOR VIDEO ACTION RECOGNITION

## Overview

We release the PyTorch code of the [Long Short temporal Network].


## Content

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Testing](#testing)
- [Training](#training)

## Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.0 or higher
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scikit-learn](https://scikit-learn.org/stable/)

For video data pre-processing, you may need [ffmpeg](https://www.ffmpeg.org/).

## Data Preparation

We need to first extract videos into frames for fast reading. Please refer to [TSN](https://github.com/yjxiong/temporal-segment-networks) repo for the detailed guide of data pre-processing.

We have successfully trained on [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/), [UCF101](http://crcv.ucf.edu/data/UCF101.php), [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), [Something-Something-V1](https://20bn.com/datasets/something-something/v1) and [V2](https://20bn.com/datasets/something-something/v2), [Jester](https://20bn.com/datasets/jester) datasets with this codebase. Basically, the processing of video data can be summarized into 3 steps:

- Extract frames from videos (refer to [tools/vid2img_kinetics.py](tools/vid2img_kinetics.py) for Kinetics example and [tools/vid2img_sthv2.py](tools/vid2img_sthv2.py) for Something-Something-V2 example)
- Generate annotations needed for dataloader (refer to [tools/gen_label_kinetics.py](tools/gen_label_kinetics.py) for Kinetics example, [tools/gen_label_sthv1.py](tools/gen_label_sthv1.py) for Something-Something-V1 example, and [tools/gen_label_sthv2.py](tools/gen_label_sthv2.py) for Something-Something-V2 example)
- Add the information to [ops/dataset_configs.py](ops/dataset_configs.py)

Note that the naive implementation involves large data copying and increases memory consumption during training. It is suggested to use the **in-place** version of TSM to improve speed (see [ops/temporal_shift.py](ops/temporal_shift.py) Line 12 for the details.)

## Testing 

For example, to test the downloaded pretrained models on Kinetics, you can run `scripts/test_tsm_kinetics_rgb_8f.sh`. The scripts will test both TSN and TSM on 8-frame setting by running:

```bash
# test TSN
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64

# test TSM
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth \
    --test_segments=8 --test_crops=1 \
    --batch_size=64
```

Change to `--test_crops=10` for 10-crop evaluation. With the above scripts, you should get around 68.8% and 71.2% results respectively.

To get the Kinetics performance of our dense sampling model under Non-local protocol, run:

```bash
# test TSN using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_avg_segment5_e50.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res

# test TSM using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res

# test NL TSM using non-local testing protocol
python test_models.py kinetics \
    --weights=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense_nl.pth \
    --test_segments=8 --test_crops=3 \
    --batch_size=8 --dense_sample --full_res
```

You should get around 70.6%, 74.1%, 75.6% top-1 accuracy, as shown in Table 1.

For the efficient (center crop and 1 clip) and accurate setting (full resolution and 2 clip) on Something-Something, you can try something like this:

```bash
# efficient setting: center crop and 1 clip
python test_models.py something \
    --weights=pretrained/TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth \
    --test_segments=8 --batch_size=72 -j 24 --test_crops=1

# accurate setting: full resolution and 2 clips (--twice sample)
python test_models.py something \
    --weights=pretrained/TSM_something_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth \
    --test_segments=8 --batch_size=72 -j 24 --test_crops=3  --twice_sample
```

## Training 

We provided several examples to train TSM with this repo:

- To train on Kinetics from ImageNet pretrained models, you can run `scripts/train_tsm_kinetics_rgb_8f.sh`, which contains:

  ```bash
  # You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
  python main.py kinetics RGB \
       --arch resnet50 --num_segments 8 \
       --gd 20 --lr 0.02 --wd 1e-4 --lr_steps 20 40 --epochs 50 \
       --batch-size 128 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres --npb
  ```

  You should get `TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth` as downloaded above. Notice that you should scale up the learning rate with batch size. For example, if you use a batch size of 256 you should set learning rate to 0.04.

- After getting the Kinetics pretrained models, we can fine-tune on other datasets using the Kinetics pretrained models. For example, we can fine-tune 8-frame Kinetics pre-trained model on UCF-101 dataset using **uniform sampling** by running:

  ```
  python main.py ucf101 RGB \
       --arch resnet50 --num_segments 8 \
       --gd 20 --lr 0.001 --lr_steps 10 20 --epochs 25 \
       --batch-size 64 -j 16 --dropout 0.8 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres \
       --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
  ```

- To train on Something-Something dataset (V1&V2), using ImageNet pre-training is usually better:

  ```bash
  python main.py something RGB \
       --arch resnet50 --num_segments 8 \
       --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 \
       --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
       --shift --shift_div=8 --shift_place=blockres --npb
  ```



