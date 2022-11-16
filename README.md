## Kaggle ELL 比赛
###代码结构
* 配置文件
    yaml风格配置文件
* 模型
    使用debertav3作为预训练模型，有基于分类的模型，也有基于回归的模型
* 数据集
* 主要代码
    * train.py 加载配置文件，获取数据集，加载模型并进行训练
    * predict_k_fold: OOF预测
  