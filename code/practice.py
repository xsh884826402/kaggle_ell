# with open('../data/old.txt') as f:
#     old = f.readlines()
# with open('../data/new.txt') as f:
#     new = f.readlines()
# old_lib = [x.split()[0] for x in old]
# new_lib = set([x.split()[0] for x in new])
# result_not_in_new = []
# # for item in old_lib:
# #     if item not in new_lib:
# #         result_not_in_new.append(item)
#
# result_not_in_new = [item for item in old_lib if item not in new_lib]
# print(result_not_in_new)


# 'lightgbm', 'lunarcalendar', openpyxl,
# import os
#
# print(os.environ.get("TOKENIZERS_PARALLELISM"))
# names = []
# with open('../data/tbnames_old.txt', encoding='utf-8') as f:
#     for line in f:
#         tb_names = line.split(',')
#         print('tb', tb_names)
#         new_tb_names = [x.replace('\\n', '').replace("'", '') for x in tb_names]
#         print('new tb', [x.replace('\\n', '').replace("'", '') for x in tb_names])
#         names.extend(new_tb_names)
# print(names)
# with open('../data/tbnames_new.txt', 'w', encoding='utf-8') as f:
#     f.write('\n'.join(names))
# import datetime
# # # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
# import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# import pickle
# print(np.argmax([1,2,4,2]))
# with open('batch_pre', 'rb') as f:
#     batch_pre = pickle.load(f)
# with open('val_y', 'rb') as f:
#     val_y = pickle.load(f)
# def cal_mse(pred, label):
#     pred = np.array(pred)
#     label = np.array(label)
#     pred = np.argmax(pred)
a = [[1, 2, 3],[1, 2, 3]]
na = np.array(a)
print(np.argmax(na, axis=-1))
print(na.transpose())
p = pd.D