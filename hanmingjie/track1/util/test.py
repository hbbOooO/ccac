
# import random

# # add dev.txt and test.txt into train,txt

# data_root_dir = '/root/autodl-tmp/data/Track2/'

# with open(data_root_dir+'train.txt') as f:
#     train_lines = f.readlines()
# with open(data_root_dir+'dev.txt') as f:
#     val_lines = f.readlines()
# with open(data_root_dir+'test.txt') as f:
#     test_lines = f.readlines()
# lines = []
# lines.extend(train_lines)
# lines.extend(val_lines)
# lines.extend(test_lines)
# # random.shuffle(lines)
# with open(data_root_dir+'mixed.txt', 'w') as f:
#     f.writelines(lines)

import pandas as pd

data_root_dir = '/root/autodl-tmp/data/qnli/'
file_names = ['train.tsv', 'dev.tsv']
out_names = ['train.txt', 'dev.txt']
out_data = []
for file_name, out_name in zip(file_names, out_names):
    # data = pd.read_csv(data_root_dir+name,skiprows=6, header=None,delimiter="\t")
    # print(data)
    with open(data_root_dir+file_name) as f:
        lines = f.readlines()[1:]
    out_lines = []
    for line in lines:
        _, point, sentence, label = line.split('\t')
        label = label[:-1]
        label = '1' if label == 'entailment' else '0'
        out_lines.append(
            point+'\t'+sentence+'\t'+label+'\n'
        )

    out_data.extend(out_lines)

with open(data_root_dir+'extra.txt', 'w') as f:
    f.writelines(out_data)