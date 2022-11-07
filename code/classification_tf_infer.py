import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, TFDebertaV2Model, AutoConfig
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import random
from sklearn.model_selection import KFold
import tensorflow_addons as tfa
import sys
sys.path.append('../input/iterativestratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from focal_loss import SparseCategoricalFocalLoss
import tensorflow_addons as tfa


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_everything(42)

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices : {strategy.num_replicas_in_sync}")
BATCH_SIZE = 4*strategy.num_replicas_in_sync
BUFFER_SIZE = 3200
AUTO = tf.data.AUTOTUNE
SEQ_LEN = 512
MODEL_NAME = "microsoft/deberta-v3-base"
CHECKPOINT_PATH = "deberta_model.h5"
CLS_NUM = 9

def preprocess_df(df):
    for col_name in ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]:
        df[col_name+"_class"] = df[col_name].map({1.0: 0, 1.5: 1, 2.0: 2, 2.5: 3, 3.0: 4, 3.5: 5, 4.0: 6, 4.5: 7, 5.0: 8})
    return df


df = pd.read_csv("../input/feedback-prize-english-language-learning/train.csv")
df = preprocess_df(df)

def preprocess(df, tokenizer):
    inputs, labels = np.array(df["full_text"]), np.array(df[["cohesion_class", "syntax_class", "vocabulary_class", "phraseology_class", "grammar_class", "conventions_class"]])
#     y = keras.utils.to_categorical(labels, CLS_NUM)
    input_ids = []
    attention_mask = []
    for x in tqdm(inputs):
        tokens = tokenizer(x, padding="max_length", truncation=True, max_length=SEQ_LEN, return_tensors="np")
        ids = tokens["input_ids"]
        mask = tokens["attention_mask"]
        input_ids.append(ids)
        attention_mask.append(mask)
    input_ids = np.array(input_ids).squeeze()
    attention_mask = np.array(attention_mask).squeeze()
    return input_ids, attention_mask, labels

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=123, shuffle=False)

train_ids, train_mask, train_y = preprocess(train_df, tokenizer)
val_ids, val_mask, val_y = preprocess(val_df, tokenizer)

TRAIN_NUM = len(train_df)
VAL_NUM = len(val_df)

def get_train_dataset(ids, mask, y):
    x = tf.data.Dataset.from_tensor_slices({
        "input_ids": tf.constant(ids, dtype="int32"),
        "attention_mask": tf.constant(mask, dtype="int32")
    })
    y = tf.data.Dataset.from_tensor_slices((
        tf.constant(y[:, 0], dtype="int32"),
        tf.constant(y[:, 1], dtype='int32'),
        tf.constant(y[:, 2], dtype='int32'),
        tf.constant(y[:, 3], dtype='int32'),
        tf.constant(y[:, 4], dtype='int32'),
        tf.constant(y[:, 5], dtype='int32'),
    ))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    data = tf.data.Dataset.zip((x, y))
    data = data.with_options(options)
    data = data.repeat()
    data = data.shuffle(BUFFER_SIZE)
    data = data.batch(BATCH_SIZE)
    data = data.prefetch(AUTO)
    return data

def get_val_dataset(ids, mask, y):
    x = tf.data.Dataset.from_tensor_slices({
        "input_ids": tf.constant(ids, dtype="int32"),
        "attention_mask": tf.constant(mask, dtype="int32")
    })
    y = tf.data.Dataset.from_tensor_slices((
        tf.constant(y[:, 0], dtype="int32"),
        tf.constant(y[:, 1], dtype='int32'),
        tf.constant(y[:, 2], dtype='int32'),
        tf.constant(y[:, 3], dtype='int32'),
        tf.constant(y[:, 4], dtype='int32'),
        tf.constant(y[:, 5], dtype='int32'),
    ))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    data = tf.data.Dataset.zip((x, y))
    data = data.with_options(options)
    data = data.repeat()
    data = data.batch(BATCH_SIZE)
    data = data.prefetch(AUTO)
    return data

train_dataset = get_train_dataset(train_ids, train_mask, train_y)
val_dataset = get_val_dataset(val_ids, val_mask, val_y)
for data in train_dataset.take(1):
    print(data)


class MeanPool(keras.layers.Layer):
    def call(self, x, mask=None):
        print('mask:', np.shape(mask))
        broad_mask = tf.cast(tf.expand_dims(mask, -1), "float32")
        # [batch, maxlen, hidden_state]
        print('broad_mask', np.shape(broad_mask))
        x = tf.math.reduce_sum( x * broad_mask, axis=1)
        x = x / tf.math.maximum(tf.reduce_sum(broad_mask, axis=1), tf.constant([1e-9]))
        return x


def build_model(trainable=True):
    input1 = keras.Input(shape=(None,), dtype="int32", name="input_ids")
    input2 = keras.Input(shape=(None,), dtype="int32", name="attention_mask")

    #     config = AutoConfig.from_pretrained(MODEL_NAME)
    #     config.attention_probs_dropout_prob = 0.0
    #     config.hidden_dropout_prob = 0.0

    base_model = TFDebertaV2Model.from_pretrained(
        MODEL_NAME,
        #         config=config,
    )
    base_model.trainable = trainable
    base_outputs = base_model.deberta({"input_ids": input1,
                                       "attention_mask": input2})
    last_hidden_state = base_outputs[0]
    x = MeanPool()(last_hidden_state, mask=input2)
    outputs = []
    # Here are 6 measures outputs of essays
    output_names = {0: "cohesion", 1: "syntax", 2: "vocabulary", 3: "phraseology", 4: "grammar", 5: "conventions"}
    for i in range(6):
        output = layers.Dense(CLS_NUM, activation="softmax", name=f"{output_names[i]}")(x)
        outputs.append(output)

    model = keras.Model(inputs={"input_ids": input1, "attention_mask": input2}, outputs=outputs)

    #     lr_schedule1 = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=5e-6,
    #                                                               decay_steps=TRAIN_NUM//BATCH_SIZE,
    #                                                               decay_rate=0.3)
    #     lr_schedule2 = keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-5,
    #                                                               decay_steps=TRAIN_NUM//BATCH_SIZE,
    #                                                               alpha=0.8)

    #     optimizers = tfa.optimizers.MultiOptimizer([(keras.optimizers.Adam(learning_rate=lr_schedule1), model.layers[:-6]),
    #                                                 (keras.optimizers.Adam(learning_rate=lr_schedule2), model.layers[-6:])])
    model.compile(
        loss=[SparseCategoricalFocalLoss(gamma=2),
              SparseCategoricalFocalLoss(gamma=2),
              SparseCategoricalFocalLoss(gamma=2),
              SparseCategoricalFocalLoss(gamma=2),
              SparseCategoricalFocalLoss(gamma=2),
              SparseCategoricalFocalLoss(gamma=2),
              ],
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        #           optimizer=optimizers,
        metrics=["accuracy"],
    )
    return model


# tf.keras.backend.clear_session()
# model = build_model(trainable=False)
# model.load_weights(CHECKPOINT_PATH)
# test_data = val_dataset.repeat(1)
# y_labels = []
# y_preds = []
#
# batch_pred = model.predict([val_ids, val_mask])
# print('batch_pred', np.shape(batch_pred))
# with open('batch_pre', 'wb') as f:
#     pickle.dump(batch_pred, f)
# print(test_data)
# pred = model.predict(val_dataset[0], batch_size=BATCH_SIZE, )
# print(type(pred), pred)
with open('batch_pre', 'rb') as f:
    batch_pre = pickle.load(f)
with open('val_y', 'wb') as f:
    pickle.dump(val_y, f)
print("type", type(batch_pre), batch_pre)


