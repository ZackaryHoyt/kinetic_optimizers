# Author: Zackary Hoyt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

data_df = pd.read_csv("outputs/data.csv")

os.makedirs("outputs/plots", exist_ok=True)

inputs = data_df[['k','c','std_classes','std_samples','d','n_samples']].to_numpy()
inputs = np.concatenate([inputs, 1 / inputs], axis=1)
inputs = np.concatenate([np.ones((len(inputs),1)), inputs], axis=1)

relative_improvement_true_iko = 100 * (data_df.loss_kmeans / data_df.loss_iko - 1)
relative_improvement_true_nko = 100 * (data_df.loss_kmeans / data_df.loss_nko - 1)

################################################################################################################################

ymin, ymax = (-2, 2)
stepsize_major, stepsize_minor = (0.5, 0.25)
major_ticks = np.arange(ymin - stepsize_major, ymax + stepsize_major, stepsize_major)
minor_ticks = np.arange(ymin - stepsize_major, ymax + stepsize_major, stepsize_minor)

# environmental_complexities = (data_df.std_samples ** 1 * data_df.c ** 1 * data_df.n_samples ** 1) ** 0.5 / (data_df.std_classes ** 1 * data_df.k ** 1 * data_df.d ** 0) ** 0.5
environmental_complexities = (data_df.c * data_df.std_samples * data_df.n_samples * data_df.d) ** 0.5 / (data_df.k * data_df.std_classes) ** 0.5
environmental_complexities -= np.min(environmental_complexities)
environmental_complexities /= np.max(environmental_complexities)

plt.clf()
plt.gcf().set_size_inches(4, 3.5)
plt.scatter(environmental_complexities, relative_improvement_true_iko, marker='.', alpha=0.1, c='black')
plt.hlines(y=0, xmin=np.min(environmental_complexities), xmax=np.max(environmental_complexities), color='w', linewidth=0.75, alpha=0.75)
plt.title("K-Means Loss Reduction (%) Analysis - IKO", fontsize=8)
plt.xlabel("Environmental Complexity Heuristic", fontsize=8)
plt.ylabel("Loss Reduction (%)", fontsize=8)
ax = plt.gca()
ax.set_xticks(ticks=major_ticks)
ax.set_xticks(ticks=minor_ticks, minor=True)
ax.set_yticks(ticks=major_ticks)
ax.set_yticks(ticks=minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
ax.set_xticklabels(minor_ticks, minor=True)
ax.tick_params(axis='both', which='both', labelsize=8)
ax.set_axisbelow(True)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(ymin, ymax)
plt.savefig("outputs/plots/kmeans_loss_reduction_analysis-iko.png", dpi=300, bbox_inches='tight', pad_inches=0.025)

plt.clf()
plt.gcf().set_size_inches(4, 3.5)
plt.scatter(environmental_complexities, relative_improvement_true_nko, marker='.', alpha=0.1, c='black')
plt.hlines(y=0, xmin=np.min(environmental_complexities), xmax=np.max(environmental_complexities), color='w', linewidth=0.75, alpha=0.75)
plt.title("K-Means Loss Reduction (%) Analysis - NKO", fontsize=8)
plt.xlabel("Environmental Complexity Heuristic", fontsize=8)
plt.ylabel("Loss Reduction (%)", fontsize=8)
ax = plt.gca()
ax.set_xticks(ticks=major_ticks)
ax.set_xticks(ticks=minor_ticks, minor=True)
ax.set_yticks(ticks=major_ticks)
ax.set_yticks(ticks=minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
ax.set_xticklabels(minor_ticks, minor=True)
ax.tick_params(axis='both', which='both', labelsize=8)
ax.set_axisbelow(True)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(ymin, ymax)
plt.savefig("outputs/plots/kmeans_loss_reduction_analysis-nko.png", dpi=300, bbox_inches='tight', pad_inches=0.025)

################################################################################################################################

model = np.array([0.182313, 0.030451, -0.026487, 0.020623, -0.016421, -0.020951, -0.006433, -0.907604, -0.085089, 0.33183, -0.182971, 0.722236, 2.513335])
y_intercept = model[0]
standardizer_mean, standardizer_std = (0.190677, 0.380321)
model_pred = standardizer_std * inputs @ model + standardizer_mean

ymin, ymax = (-2, 2)
stepsize_major, stepsize_minor = (0.5, 1/8)
major_ticks = np.arange(ymin - stepsize_major, ymax + stepsize_major, stepsize_major)
minor_ticks = np.arange(ymin - stepsize_major, ymax + stepsize_major, stepsize_minor)

plt.clf()
plt.gcf().set_size_inches(4, 3.5)
plt.scatter(model_pred, relative_improvement_true_iko, marker='.', alpha=0.1, c='black')
plt.hlines(y=0, xmin=np.min(model_pred), xmax=np.max(model_pred), color='w', linewidth=0.75, alpha=0.75)
plt.vlines(x=0, ymin=np.min(relative_improvement_true_iko), ymax=np.max(relative_improvement_true_iko), color='w', linewidth=0.75, alpha=0.75)
plt.plot(model_pred, np.poly1d(np.polyfit(model_pred, relative_improvement_true_iko, 1))(model_pred), color='r', linewidth=1, alpha=0.8)
plt.title("K-Means Loss Reduction (%) Model - IKO", fontsize=8)
plt.xlabel("Predicted (%)", fontsize=8)
plt.ylabel("Actual (%)", fontsize=8)
ax = plt.gca()
ax.set_xticks(ticks=major_ticks)
ax.set_xticks(ticks=minor_ticks, minor=True)
ax.set_yticks(ticks=major_ticks)
ax.set_yticks(ticks=minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
ax.set_xticklabels(minor_ticks, minor=True)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_axisbelow(True)
ax.set_xlim(-0.25, 0.75)
ax.set_ylim(ymin, ymax)
plt.savefig("outputs/plots/kmeans_loss_reduction_model-iko.png", dpi=300, bbox_inches='tight', pad_inches=0.025)

model = np.array([-1.422225, 0.018261, 0.043682, 0.031455, 0.027509, -0.000263, 0.004073, 0.355122, 0.549332, 0.788283, 0.528135, -0.192229, 1.016112])
y_intercept = model[0]
standardizer_mean, standardizer_std = (-0.021821, 0.243580)
model_pred = standardizer_std * inputs @ model + standardizer_mean

stepsize_major, stepsize_minor = (0.5, 1/16)
major_ticks = np.arange(ymin - stepsize_major, ymax + stepsize_major, stepsize_major)
minor_ticks = np.arange(ymin - stepsize_major, ymax + stepsize_major, stepsize_minor)

plt.clf()
plt.gcf().set_size_inches(4, 3.5)
plt.scatter(model_pred, relative_improvement_true_nko, marker='.', alpha=0.1, c='black')
plt.hlines(y=0, xmin=np.min(model_pred), xmax=np.max(model_pred), color='w', linewidth=0.75, alpha=0.75)
plt.vlines(x=0, ymin=np.min(relative_improvement_true_nko), ymax=np.max(relative_improvement_true_nko), color='w', linewidth=0.75, alpha=0.75)
plt.plot(model_pred, np.poly1d(np.polyfit(model_pred, relative_improvement_true_nko, 1))(model_pred), color='r', linewidth=1, alpha=0.8)
plt.title("K-Means Loss Reduction (%) Model - NKO", fontsize=8)
plt.xlabel("Predicted (%)", fontsize=8)
plt.ylabel("Actual (%)", fontsize=8)
ax = plt.gca()
ax.set_xticks(ticks=major_ticks)
ax.set_xticks(ticks=minor_ticks, minor=True)
ax.set_yticks(ticks=major_ticks)
ax.set_yticks(ticks=minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
ax.set_xticklabels(minor_ticks, minor=True)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='both', which='major', labelsize=8)
ax.set_axisbelow(True)
ax.set_xlim(-0.125, 0.0625)
ax.set_ylim(ymin, ymax)
plt.savefig("outputs/plots/kmeans_loss_reduction_model-nko.png", dpi=300, bbox_inches='tight', pad_inches=0.025)

################################################################################################################################

ymin, ymax = (0, 25)
stepsize_major, stepsize_minor = (5, 0.25)
major_ticks = np.arange(ymin - stepsize_major, ymax + stepsize_major, stepsize_major)
minor_ticks = np.arange(ymin - stepsize_major, ymax + stepsize_major, stepsize_minor)

solution_complexities = data_df.c ** 0.5 * data_df.n_samples ** 0.5# / (data_df.std_classes ** 0.5 / data_df.std_samples ** 0.5) ** 0.5
solution_complexities /= np.max(solution_complexities)

plt.clf()
plt.gcf().set_size_inches(4, 3.5)
plt.scatter(solution_complexities, data_df.n_gens_kmeans, marker='.', alpha=0.1, c='black')
plt.title("Convergence Speed (# of Steps) Analysis - K-Means", fontsize=8)
plt.xlabel("Solution Complexity Heuristic", fontsize=8)
plt.ylabel("Convergence Speed (# of Steps)", fontsize=8)
ax = plt.gca()
ax.set_xticks(ticks=major_ticks)
ax.set_xticks(ticks=minor_ticks, minor=True)
ax.set_yticks(ticks=major_ticks)
ax.set_yticks(ticks=minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
ax.set_xticklabels(minor_ticks, minor=True)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='both', which='both', labelsize=8)
ax.set_axisbelow(True)
ax.set_xlim(0, 1.125)
ax.set_ylim(ymin, ymax)
plt.savefig("outputs/plots/convergence_speed_analysis-kmeans.png", dpi=300, bbox_inches='tight', pad_inches=0.025)

plt.clf()
plt.gcf().set_size_inches(4, 3.5)
plt.scatter(solution_complexities, data_df.n_gens_iko, marker='.', alpha=0.1, c='black')
plt.title("Convergence Speed (# of Steps) Analysis - IKO", fontsize=8)
plt.xlabel("Solution Complexity Heuristic", fontsize=8)
plt.ylabel("Convergence Speed (# of Steps)", fontsize=8)
ax = plt.gca()
ax.set_xticks(ticks=major_ticks)
ax.set_xticks(ticks=minor_ticks, minor=True)
ax.set_yticks(ticks=major_ticks)
ax.set_yticks(ticks=minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
ax.set_xticklabels(minor_ticks, minor=True)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='both', which='both', labelsize=8)
ax.set_axisbelow(True)
ax.set_xlim(0, 1.125)
ax.set_ylim(ymin, ymax)
plt.savefig("outputs/plots/convergence_speed_analysis-iko.png", dpi=300, bbox_inches='tight', pad_inches=0.025)

plt.clf()
plt.gcf().set_size_inches(4, 3.5)
plt.scatter(solution_complexities, data_df.n_gens_nko, marker='.', alpha=0.1, c='black')
plt.title("Convergence Speed (# of Steps) Analysis - NKO", fontsize=8)
plt.xlabel("Solution Complexity Heuristic", fontsize=8)
plt.ylabel("Convergence Speed (# of Steps)", fontsize=8)
ax = plt.gca()
ax.set_xticks(ticks=major_ticks)
ax.set_xticks(ticks=minor_ticks, minor=True)
ax.set_yticks(ticks=major_ticks)
ax.set_yticks(ticks=minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
ax.set_xticklabels(minor_ticks, minor=True)
ax.tick_params(axis='x', which='minor', labelsize=6)
ax.tick_params(axis='both', which='both', labelsize=8)
ax.set_axisbelow(True)
ax.set_xlim(0, 1.125)
ax.set_ylim(ymin, ymax)
plt.savefig("outputs/plots/convergence_speed_analysis-nko.png", dpi=300, bbox_inches='tight', pad_inches=0.025)

################################################################################################################################

ymin, ymax = (-5, 25)
stepsize_major, stepsize_minor = (5, 1)
major_ticks = np.arange(ymin - stepsize_major, ymax + stepsize_major, stepsize_major)
minor_ticks = np.arange(ymin - stepsize_major, ymax + stepsize_major, stepsize_minor)

model = np.array([-0.120153, -0.066979, 0.164466, -0.048893, 0.023457, -0.029914, 0.025264, -1.806085, -0.555891, 0.304652, -1.036519, 0.169387, -3.457291])
y_intercept = model[0]
standardizer_mean, standardizer_std = (8.709561, 4.546382)
model_pred = standardizer_std * inputs @ model + standardizer_mean

plt.clf()
plt.gcf().set_size_inches(4, 3.5)
plt.scatter(model_pred, data_df.n_gens_kmeans, marker='.', alpha=0.1, c='black')
plt.plot(model_pred, np.poly1d(np.polyfit(model_pred, data_df.n_gens_kmeans, 1))(model_pred), color='r', linewidth=1, alpha=0.8)
plt.title("Convergence Speed (# of Steps) Model - K-Means", fontsize=8)
plt.xlabel("Predicted (# of Steps)", fontsize=8)
plt.ylabel("Actual (# of Steps)", fontsize=8)
ax = plt.gca()
ax.set_xticks(ticks=major_ticks)
ax.set_xticks(ticks=minor_ticks, minor=True)
ax.set_yticks(ticks=major_ticks)
ax.set_yticks(ticks=minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
ax.tick_params(axis='both', which='both', labelsize=8)
ax.set_axisbelow(True)
ax.set_xlim(ymin, ymax)
ax.set_ylim(ymin, ymax)
plt.gca().set_aspect("equal")
plt.savefig("outputs/plots/convergence_speed_model-kmeans.png", dpi=300, bbox_inches='tight', pad_inches=0.025)

model = np.array([0.015944, -0.042284, 0.151034, -0.047985, 0.02793, -0.034581, 0.021721, -1.867263, -0.635355, 0.472226, -1.059437, -0.020857, -4.284342])
y_intercept = model[0]
standardizer_mean, standardizer_std = (8.071080, 3.631180)
model_pred = standardizer_std * inputs @ model + standardizer_mean

plt.clf()
plt.gcf().set_size_inches(4, 3.5)
plt.scatter(model_pred, data_df.n_gens_iko, marker='.', alpha=0.1, c='black')
plt.plot(model_pred, np.poly1d(np.polyfit(model_pred, data_df.n_gens_iko, 1))(model_pred), color='r', linewidth=1, alpha=0.8)
plt.title("Convergence Speed (# of Steps) Model - IKO", fontsize=8)
plt.xlabel("Predicted (# of Steps)", fontsize=8)
plt.ylabel("Actual (# of Steps)", fontsize=8)
ax = plt.gca()
ax.set_xticks(ticks=major_ticks)
ax.set_xticks(ticks=minor_ticks, minor=True)
ax.set_yticks(ticks=major_ticks)
ax.set_yticks(ticks=minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
ax.tick_params(axis='both', which='both', labelsize=8)
ax.set_axisbelow(True)
ax.set_xlim(ymin, ymax)
ax.set_ylim(ymin, ymax)
plt.gca().set_aspect("equal")
plt.savefig("outputs/plots/convergence_speed_model-iko.png", dpi=300, bbox_inches='tight', pad_inches=0.025)

model = np.array([-0.19284, -0.067348, 0.16672, -0.048953, 0.027088, -0.029302, 0.025603, -1.861217, -0.499491, 0.307677, -0.946968, 0.150215, -3.319022])
y_intercept = model[0]
standardizer_mean, standardizer_std = (8.878671, 4.758094)
model_pred = standardizer_std * inputs @ model + standardizer_mean

plt.clf()
plt.gcf().set_size_inches(4, 3.5)
plt.scatter(model_pred, data_df.n_gens_nko, marker='.', alpha=0.1, c='black')
plt.plot(model_pred, np.poly1d(np.polyfit(model_pred, data_df.n_gens_nko, 1))(model_pred), color='r', linewidth=1, alpha=0.8)
plt.title("Convergence Speed (# of Steps) Model - NKO", fontsize=8)
plt.xlabel("Predicted (# of Steps)", fontsize=8)
plt.ylabel("Actual (# of Steps)", fontsize=8)
ax = plt.gca()
ax.set_xticks(ticks=major_ticks)
ax.set_xticks(ticks=minor_ticks, minor=True)
ax.set_yticks(ticks=major_ticks)
ax.set_yticks(ticks=minor_ticks, minor=True)
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.8)
ax.tick_params(axis='both', which='both', labelsize=8)
ax.set_axisbelow(True)
ax.set_xlim(ymin, ymax)
ax.set_ylim(ymin, ymax)
plt.gca().set_aspect("equal")
plt.savefig("outputs/plots/convergence_speed_model-nko.png", dpi=300, bbox_inches='tight', pad_inches=0.025)
