from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores
df_final = pd.read_csv("output/classification/predicts.csv")
oof_score = get_score(y_trues=df_final[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].values,
                      y_preds=df_final[[item + '_label' for item in ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']]])
print(f'oof_score: {oof_score}')