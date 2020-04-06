import pandas as pd
import numpy as np
import sys
# import collections
# import csv
# import os
# import modeling
# import optimization
# import tokenization
import tensorflow as tf

sys.path.append('../../..')

from ml import get_model_dir, get_log_dir  # noqa
from ml.datasets import get_data_path  # noqa

#################################
# CONFIG
#################################
weibo_path = get_data_path('weibo')
model_name = 'bert'
experiment_name = 'weibo'
model_dir = get_model_dir()
log_dir = get_log_dir(model_name, experiment_name)
model_path = model_dir / f'{model_name}_{experiment_name}.h5'

validation_rate = 0.1

# other param
bert_config_file = None
vocab_file = None
output_dir = None
init_checkpoint = None
max_seq_length = 128
do_train = False
do_eval = False
do_predict = False
train_batch_size = 32
eval_batch_size = 8
predict_batch_size = 8
learning_rate = 5e-5
num_train_epochs = 3
warmup_proportion = 0.1
save_checkpoints_steps = 1000
iterations_per_loop = 1000


def prepare():
    weibo = pd.read_csv(weibo_path)
    weibo.sample(frac=1)
    msk = np.random.rand(len(weibo)) < VAL_RATE
    data_val = weibo[msk]
    data_train = weibo[~msk]
    x_train, y_train = data_train.review, data_train.label
    x_val, y_val = data_val.review, data_val.label
