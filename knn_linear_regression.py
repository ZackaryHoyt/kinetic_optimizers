# Author: Zackary Hoyt

import pandas as pd
import numpy as np
np.random.seed(0)

training_partition_size = 0.5

# Load Data
data_df = pd.read_csv("outputs/data.csv")
inputs = data_df[['k','c','std_classes','std_samples','d','n_samples']].to_numpy()
inputs = np.concatenate([inputs, 1 / inputs], axis=1)
# inputs = np.concatenate([inputs, inputs ** 2], axis=1)
# inputs = np.concatenate([inputs, inputs ** 0.5], axis=1)
inputs = np.concatenate([np.ones((len(inputs),1)), inputs], axis=1)
[n, p] = inputs.shape

# Create Train & Test Partitions
n_train = int(training_partition_size * n)
n_test = n - n_train
idx = np.random.permutation(n)
idx_train = idx[0:n_train]
idx_test = idx[n_train:]

sample_train = inputs[idx_train]
sample_test = inputs[idx_test]

def ls_regression(labels_train, labels_test):
	# Define the Model
	model = np.linalg.inv(sample_train.transpose() @ sample_train) @ (sample_train.transpose() @ labels_train)

	# Train Data Evaluation
	pred_train = np.matmul(sample_train, model)
	err_train = np.sqrt(np.mean((labels_train - pred_train) ** 2) / n_train)

	# Test Data Evaluation
	pred_test = np.matmul(sample_test, model)
	err_test = np.sqrt(np.mean((labels_test - pred_test) ** 2) / n_test)
	
	return (model, err_train, err_test)

labels = 100 * (data_df.loss_kmeans / data_df.loss_iko - 1).to_numpy()
labels_train, labels_test = (labels[idx_train], labels[idx_test])
sample_mean, sample_std = (np.average(labels_train), np.std(labels_train))
(model, err_train, err_test) = ls_regression(labels_train=(labels_train - sample_mean) / sample_std, labels_test=(labels_test - sample_mean) / sample_std)
print("K-Means Loss Reduction - IKO")
print(model.round(6).tolist())
print("Training RMSE = {0:.06f}".format(err_train))
print("Testing RMSE = {0:.06f}".format(err_test))
print("Training Label Avg = {0:.06f}, Training Label Std = {1:.06f}".format(sample_mean, sample_std))
print("====================================")

labels = 100 * (data_df.loss_kmeans / data_df.loss_nko - 1).to_numpy()
labels_train, labels_test = (labels[idx_train], labels[idx_test])
sample_mean, sample_std = (np.average(labels_train), np.std(labels_train))
(model, err_train, err_test) = ls_regression(labels_train=(labels_train - sample_mean) / sample_std, labels_test=(labels_test - sample_mean) / sample_std)
print("K-Means Loss Reduction - NKO")
print(model.round(6).tolist())
print("Training RMSE = {0:.06f}".format(err_train))
print("Testing RMSE = {0:.06f}".format(err_test))
print("Training Label Avg = {0:.06f}, Training Label Std = {1:.06f}".format(sample_mean, sample_std))
print("====================================")

labels=data_df.n_gens_kmeans.to_numpy()
labels_train, labels_test = (labels[idx_train], labels[idx_test])
sample_mean, sample_std = (np.average(labels_train), np.std(labels_train))
(model, err_train, err_test) = ls_regression(labels_train=(labels_train - sample_mean) / sample_std, labels_test=(labels_test - sample_mean) / sample_std)
print("# Steps Required - K-Means")
print(model.round(6).tolist())
print("Training RMSE = {0:.06f}".format(err_train))
print("Testing RMSE = {0:.06f}".format(err_test))
print("Training Label Avg = {0:.06f}, Training Label Std = {1:.06f}".format(sample_mean, sample_std))
print("====================================")

labels=data_df.n_gens_iko.to_numpy()
labels_train, labels_test = (labels[idx_train], labels[idx_test])
sample_mean, sample_std = (np.average(labels_train), np.std(labels_train))
(model, err_train, err_test) = ls_regression(labels_train=(labels_train - sample_mean) / sample_std, labels_test=(labels_test - sample_mean) / sample_std)
print("# Steps Required - IKO")
print(model.round(6).tolist())
print("Training RMSE = {0:.06f}".format(err_train))
print("Testing RMSE = {0:.06f}".format(err_test))
print("Training Label Avg = {0:.06f}, Training Label Std = {1:.06f}".format(sample_mean, sample_std))
print("====================================")

labels=data_df.n_gens_nko.to_numpy()
labels_train, labels_test = (labels[idx_train], labels[idx_test])
sample_mean, sample_std = (np.average(labels_train), np.std(labels_train))
(model, err_train, err_test) = ls_regression(labels_train=(labels_train - sample_mean) / sample_std, labels_test=(labels_test - sample_mean) / sample_std)
print("# Steps Required - NKO")
print(model.round(6).tolist())
print("Training RMSE = {0:.06f}".format(err_train))
print("Testing RMSE = {0:.06f}".format(err_test))
print("Training Label Avg = {0:.06f}, Training Label Std = {1:.06f}".format(sample_mean, sample_std))
print("====================================")
