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
# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
import torch
x1 = torch.tensor([[11, 21, 31], [21, 31, 41]], dtype=torch.int)
x2 = torch.tensor([[12, 22, 32], [22, 32, 42]])


x3 = torch.cat([x1, x2], dim=len(x1.shape)-1)
print(x1.unsqueeze(2))
print(x1.unsqueeze(2).shape)
print(x1.unsqueeze(2).squeeze(2))
print(x1.unsqueeze(2).squeeze(2).shape)