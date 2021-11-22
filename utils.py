import torch
import random
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, matthews_corrcoef

# Calculate performance metric
def calculate_metric(y_true, y_pred, y_pred_proba):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    mcc = matthews_corrcoef(y_true, y_pred)

    return acc, f1, auc, mcc

# Check mean and standard deviation
def check_mean_std_performance(result):
    return_list = []

    for m in ['ACC', 'F1', 'AUC', 'MCC']:
        return_list.append('{:.2f}+-{:.2f}'.format(np.array(result[m]).mean() * 100, np.array(result[m]).std() * 100))

    return return_list

# Setting random seed
def set_seed(random_seed):
    # Seed Setting
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# For early stopping
class EarlyStopping:
    def __init__(self, patience=100, delta=1e-3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        elif score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.early_stop = False

        else:
            self.best_score = score
            self.counter = 0

# Toy Dataset Class
"""
- Label: Binary
- Modality1: n(376) x d1 (18164)
- Modality1: n(376) x d2 (19353)
- Modality1: n(376) x d3 (309)
"""
class Toy_Dataset:
    def __init__(self, random_seed):
        # Make random dataset
        label = np.random.randint(2, size=(376, 1))
        data1 = np.random.rand(376, 18164)
        data2 = np.random.rand(376, 19353)
        data3 = np.random.rand(376, 309)

        # 5CV Dataset
        self.dataset = {'cv1': None, 'cv2': None, 'cv3': None, 'cv4': None, 'cv5': None}

        # Split Train,Validation and Test with 5 CV Fold
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        for i, (train_val_index, test_index) in enumerate(kf.split(data1, label)):
            x_train_val_1, x_test_1, y_train_val, y_test = data1[train_val_index], data1[test_index], label[
                train_val_index], label[test_index]
            x_train_val_2, x_test_2 = data2[train_val_index], data2[test_index]
            x_train_val_3, x_test_3 = data3[train_val_index], data3[test_index]

            # Split Train and Validation
            x_train_1, x_val_1, y_train, y_val = train_test_split(x_train_val_1, y_train_val, test_size=0.2,
                                                                  random_state=random_seed)
            x_train_2, x_val_2, _, _ = train_test_split(x_train_val_2, y_train_val, test_size=0.2,
                                                        random_state=random_seed)
            x_train_3, x_val_3, _, _ = train_test_split(x_train_val_3, y_train_val, test_size=0.2,
                                                        random_state=random_seed)

            # CV Dataset
            cv_dataset = [[x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2],
                          [x_train_3, x_val_3, x_test_3], [y_train, y_val, y_test]]
            self.dataset['cv' + str(i + 1)] = cv_dataset

    def __call__(self, cv, tensor=True, device=None):
        [x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2], [x_train_3, x_val_3, x_test_3], \
        [y_train, y_val, y_test] = self.dataset['cv' + str(cv + 1)]

        # Numpy to tensor
        # Modality 1
        x_train_1 = torch.tensor(x_train_1).float().to(device)
        x_val_1 = torch.tensor(x_val_1).float().to(device)
        x_test_1 = torch.tensor(x_test_1).float().to(device)

        # Modality 2
        x_train_2 = torch.tensor(x_train_2).float().to(device)
        x_val_2 = torch.tensor(x_val_2).float().to(device)
        x_test_2 = torch.tensor(x_test_2).float().to(device)

        # Modality 3
        x_train_3 = torch.tensor(x_train_3).float().to(device)
        x_val_3 = torch.tensor(x_val_3).float().to(device)
        x_test_3 = torch.tensor(x_test_3).float().to(device)

        # Label
        y_train = torch.tensor(y_train).long().to(device)
        y_val = torch.tensor(y_val).long().to(device)
        y_test = torch.tensor(y_test).long().to(device)

        return [x_train_1, x_val_1, x_test_1], [x_train_2, x_val_2, x_test_2], [x_train_3, x_val_3, x_test_3], \
        [y_train, y_val, y_test]