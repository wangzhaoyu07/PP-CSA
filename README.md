# PP-CSA: Practical Privacy-Preserving Software Call Stack Analysis

## Description

Code accompanying "PP-CSA: Practical Privacy-Preserving Software Call Stack Analysis" paper (https://dl.acm.org/doi/pdf/10.1145/3649856).

## Installation

```bash
$ pip install -r requirements.txt
```
## Usage

First, download the android datasets:
```bash
$ cd assets
$ wget https://github.com/presto-osu/ecoop21/releases/download/dataset/traces.tar.gz
$ tar -xzvf traces.tar.gz 
Then rename the folder to android.
```

Then, run the following commands to initialize:

```bash
python script_for_initialization.py
```

After that, you can run the following commands to run PP-CSA:

Run for java datasets:

```bash
python main.py \
      --dataset {dataset_name} \
      --name {exp_name} \
      --seed 42 \
      --output_dir output/{dataset_name}  \
      --asset_dir assets/java \
      --custom_train_path assets/java/{dataset_name}/{dataset_name}_enter_exit_10000.csv \
      --train_ratio 0.95 \
      --data_split_cutoff 10000 \
      --epochs 10 \
      --batch_size 64 \
      --learning_rate 0.01 \
      --optim_type adam \
      --hidden_size 128 \
      --max_seq_len 20 \
      --embed_size 300 \
      --private True \
      --epsilon 10 \
      --delta 1e-05 \
      --dp_mechanism gaussian \
      --dp_module clip_norm \
      --clipping_constant 5 \
      --max_beam_size 3 \
      --use_call_matrix True \
      --is_pruning True \
      --use_cfg True \
      --is_pruning_r2 True \
      --is_android False \
```

Run for android datasets:

```bash
python main.py \
      --dataset {dataset_name} \
      --name {exp_name} \
      --seed 42 \
      --output_dir output/{dataset_name}  \
      --asset_dir assets/android \
      --custom_train_path assets/android/{dataset_name}/list \
      --train_ratio 0.95 \
      --data_split_cutoff 10000 \
      --epochs 10 \
      --batch_size 64 \
      --learning_rate 0.01 \
      --optim_type adam \
      --hidden_size 128 \
      --max_seq_len 20 \
      --embed_size 300 \
      --private True \
      --epsilon 10 \
      --delta 1e-05 \
      --dp_mechanism gaussian \
      --dp_module clip_norm \
      --clipping_constant 5 \
      --max_beam_size 3 \
      --use_call_matrix True \
      --is_pruning True \
      --use_cfg True \
      --is_pruning_r2 True \
      --is_android True \
```