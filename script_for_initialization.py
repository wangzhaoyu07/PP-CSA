import subprocess
import pandas as pd
import numpy as np
import os

def restore_sequences_for_android(s):
        stack = []
        sequences = set()
        actions = s#.split(";")
        for action in actions:
            action = action.strip()
            if len(sequences)>=50:
                break
            if action.startswith("E-"):
                item = action[2:]
                stack.append(item)
                if stack:
                    # 栈不为空，保存一条序列
                    sequences.add(';'.join(stack))

            elif action.startswith("X-"):
                item = action[2:]
                if stack and stack[-1] == item:
                    stack.pop()
                else:
                    # 栈中不存在该元素
                    return []
            else:
                # 非法输入
                return []
        return sorted(list(sequences))

dataset_name_ls =['barometer','bible','dpm','drumpads','equibase','moonphases',
                   'localtv','loctracker','parking',
                    'mitula','parrot','post','quicknews','vidanta','speedlogic']

assets_dir = "place_holder/assets" # change place_holder to your own path
output_dir = "place_holder/output" # change place_holder to your own path

# initialize the r2 threshold for each dataset
for file_name in dataset_name_ls:
    print("file_name:",file_name)
    path = f'{assets_dir}/android/{file_name}/list'
    with open(path, 'r') as f:
        f_paths = f.readlines()
    traces = []
    f_dir = f'{assets_dir}/android/{file_name}'
    f_count=0
    total_file_num = len(f_paths)
    while f_count < 1000:
        with open(os.path.join(f_dir,f_paths[f_count%total_file_num].strip()), 'r') as f:
            data = f.readlines()
            traces+=restore_sequences_for_android(data)
        f_count+=1
    df_exploded = pd.DataFrame({'sub_trace': traces})

    df_exploded.columns=["label"]
    df_exploded['text'] = np.array([None] * len(df_exploded))
    if np.all(np.array(df_exploded['text']) == None):
        print("yes")
    df_exploded['len']=df_exploded['label'].apply(lambda x: len(x.split(";")))
    print(len(list(set(df_exploded['label']))))
    tipmean=df_exploded['len'].mean()
    tipstd = df_exploded['len'].std()
    topnum1 =tipmean+3*tipstd
    with open(f'{assets_dir}/android/{file_name}/threshold', 'w') as f:
        f.write(str(int(topnum1)))

for dataset_name in dataset_name_ls:
    # without cg, matrix feature, and pruning
    command = f'python init.py \
    --dataset {dataset_name} \
    --custom_train_path "{assets_dir}/android/{dataset_name}/list" \
    --name init \
    --train_ratio 0.95 \
    --data_split_cutoff 10000 \
    --max_seq_len 20 \
    --seed 42 \
    --output_dir "{output_dir}/{dataset_name}/"  \
    --asset_dir "{assets_dir}/android" \
    --use_call_matrix False \
    --where_matrix 1 \
    --matrix_type 0 \
    --is_pruning False \
    --use_cfg False \
    --is_pruning_r2 False \
    --is_android True \
    '
    print(command)
    subprocess.call(command, shell=True)