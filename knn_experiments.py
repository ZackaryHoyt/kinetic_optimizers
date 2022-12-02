# Author: Zackary Hoyt

import numpy as np
np.seterr(all='raise')
from copy import deepcopy

"""
Allocates the given items to a cluster by the index of the given centres. Items are allocated to the first cluster closest to
them.
Returns a tuple containing the generated clusters and their corresponding mean absolute errors.
"""
def generate_clusters(centres, items) -> tuple:
	clusters = [[] for _ in centres]
	cluster_losses = [0 for _ in centres]
	for item in items:
		selected_cluster_index = None
		selected_cluster_error = np.inf
		for i in range(len(clusters)):
			cluster_error = np.linalg.norm(centres[i] - item, ord=2)
			if cluster_error < selected_cluster_error:
				selected_cluster_index = i
				selected_cluster_error = cluster_error
		clusters[selected_cluster_index].append(item)
		cluster_losses[selected_cluster_index] += selected_cluster_error
	cluster_losses = [(loss / size) if size else 0 for loss, size in zip(cluster_losses, [len(cluster) for cluster in clusters])]
	return (clusters, cluster_losses)

"""
Gradient function.
Experimented with a number of variations and found this design, which has the highest likelihood to be better than the k-means
clustering algorithm. Notably, the algorithm starts at 0% on the first timestep as there is no previous loss to evaluate it
against. This means the first update is simply the averaged position of the clusters, like the k-means algorithm. However, this
gradient will then jump to ~30% and then climb to ~50% as the optimizer converges onto a solution.
Returns the calculated gradient.
"""
def f_Δ(curr_loss, prev_loss) -> float:
	return (curr_loss / (curr_loss + prev_loss)) if prev_loss > 0 else 0

"""
Ideal Kinetic Update (IKU) algorithm implementation.
Returns the updated model state.
"""
def iku(x_1, x_2, Δ) -> np.ndarray:
	if Δ < 0:
		return iku(x_1=x_2, x_2=x_1, Δ=-Δ)
	return x_1 + Δ * (x_1 - x_2)

"""
Nonideal Kinetic Update (NKU) algorithm implementation.
Returns the updated model state.
"""
def nku(x_1, x_2, Δ, α, β) -> np.ndarray:
	if Δ < 0:
		return nku(x_1=x_2, x_2=x_1, Δ=-Δ, α=α, β=β)
	Γ = [β if x_1_i >= x_2_i else α for x_1_i, x_2_i in zip(x_1, x_2)]
	return x_1 + (Δ / (β - α)) * (Γ - x_1) * (x_1 - x_2)

"""
Runs the k-means clustering algorithm over the given items, using k clusters initialized to the given indices.
Returns the history of losses for training the model.
"""
def run_kmeans(init_indices, items, k) -> list:
	centres = [np.array(items[init_index]) for init_index in init_indices]
	centres_next = deepcopy(centres)
	losses = []
	while True:
		clusters, cluster_losses = generate_clusters(centres=centres, items=items)
		centres_next = [np.average(clusters[i], axis=0) if clusters[i] else centres_next[i] for i in range(k)]
		losses.append(np.average(cluster_losses))
		if all([np.linalg.norm(nxt - curr, ord=2) == 0 for nxt, curr in zip(centres_next, centres)]):
			return losses
		centres = deepcopy(centres_next)

"""
Runs the IKO clustering algorithm over the given items, using k clusters initialized to the given indices. The gradient
function takes 2 arguments: current cluster loss and previous cluster loss, in that order. Additionally, a patience threshold
is used to shortcircuit the training once improvements are sufficiently minimal.
Returns the history of losses for training the model.
"""
def run_iko_loss_timeout(init_indices, items, k, f_Δ, min_loss) -> list:
	centres = [np.array(items[init_index]) for init_index in init_indices]
	centres_next = deepcopy(centres)
	losses = []
	prev_cluster_losses = [np.inf for _ in range(k)]
	while True:
		clusters, cluster_losses = generate_clusters(centres=centres, items=items)
		centres_next = [iku(
				x_1=np.average(clusters[i], axis=0), x_2=centres[i],
				Δ=f_Δ(curr_loss=cluster_losses[i], prev_loss=prev_cluster_losses[i])
			) if clusters[i] else centres_next[i] for i in range(k)]
		if all((np.abs(np.subtract(prev_cluster_losses, cluster_losses))) <= min_loss):
			loss = np.average(cluster_losses)
			if loss < losses[-1]:
				losses.append(loss)
			return losses
		losses.append(np.average(cluster_losses))
		prev_cluster_losses = cluster_losses
		centres = deepcopy(centres_next)

"""
Runs the NKO clustering algorithm over the given items, using k clusters initialized to the given indices. The gradient
function takes 2 arguments: current cluster loss and previous cluster loss, in that order. Additionally, a patience threshold
is used to shortcircuit the training once improvements are sufficiently minimal.
Returns the history of losses for training the model.
"""
def run_nko_loss_timeout(init_indices, items, k, f_Δ, min_loss, α, β) -> list:
	centres = [np.array(items[init_index]) for init_index in init_indices]
	centres_next = deepcopy(centres)
	losses = []
	prev_cluster_losses = [np.inf for _ in range(k)]
	while True:
		clusters, cluster_losses = generate_clusters(centres=centres, items=items)
		centres_next = [nku(
				x_1=np.average(clusters[i], axis=0), x_2=centres[i],
				Δ=f_Δ(curr_loss=cluster_losses[i], prev_loss=prev_cluster_losses[i]),
				α=α, β=β
			) if clusters[i] else centres_next[i] for i in range(k)]
		if all((np.abs(np.subtract(prev_cluster_losses, cluster_losses))) <= min_loss):
			loss = np.average(cluster_losses)
			if loss < losses[-1]:
				losses.append(loss)
			return losses
		losses.append(np.average(cluster_losses))
		prev_cluster_losses = cluster_losses
		centres = deepcopy(centres_next)

"""
Runs the IKO clustering algorithm over the given items, using k clusters initialized to the given indices. The gradient
function takes 2 arguments: current cluster loss and previous cluster loss, in that order. Additionally, a patience threshold
is used to shortcircuit the training once the next cluster centers are sufficiently similar to the current.
Returns the history of losses for training the model.
"""
def run_iko_dist_timeout(init_indices, items, k, f_Δ, min_dist) -> list:
	centres = [np.array(items[init_index]) for init_index in init_indices]
	centres_next = deepcopy(centres)
	losses = []
	prev_cluster_losses = [np.inf for _ in range(k)]
	while True:
		clusters, cluster_losses = generate_clusters(centres=centres, items=items)
		centres_next = [iku(
				x_1=np.average(clusters[i], axis=0), x_2=centres[i],
				Δ=f_Δ(curr_loss=cluster_losses[i], prev_loss=prev_cluster_losses[i])
			) if clusters[i] else centres_next[i] for i in range(k)]
		if all([np.linalg.norm(nxt - curr, ord=2) <= min_dist for nxt, curr in zip(centres_next, centres)]):
			# Since the algorithm is not guaranteed to be monotonically decreasing in terms of loss, the final step can be
			# ignored if it is worse than the current.
			loss = np.average(cluster_losses)
			if loss < losses[-1]:
				losses.append(loss)
			return losses
		losses.append(np.average(cluster_losses))
		prev_cluster_losses = cluster_losses
		centres = deepcopy(centres_next)

"""
Runs the NKO clustering algorithm over the given items, using k clusters initialized to the given indices. The gradient
function takes 2 arguments: current cluster loss and previous cluster loss, in that order. Additionally, a patience threshold
is used to shortcircuit the training once the next cluster centers are sufficiently similar to the current.
Returns the history of losses for training the model.
"""
def run_nko_dist_timeout(init_indices, items, k, f_Δ, min_dist, α, β) -> list:
	centres = [np.array(items[init_index]) for init_index in init_indices]
	centres_next = deepcopy(centres)
	losses = []
	prev_cluster_losses = [np.inf for _ in range(k)]
	while True:
		clusters, cluster_losses = generate_clusters(centres=centres, items=items)
		centres_next = [nku(
				x_1=np.average(clusters[i], axis=0), x_2=centres[i],
				Δ=f_Δ(curr_loss=cluster_losses[i], prev_loss=prev_cluster_losses[i]),
				α=α, β=β
			) if clusters[i] else centres_next[i] for i in range(k)]
		if all([np.linalg.norm(nxt - curr, ord=2) <= min_dist for nxt, curr in zip(centres_next, centres)]):
			# Since the algorithm is not guaranteed to be monotonically decreasing in terms of loss, the final step can be
			# ignored if it is worse than the current.
			loss = np.average(cluster_losses)
			if loss < losses[-1]:
				losses.append(loss)
			return losses
		losses.append(np.average(cluster_losses))
		prev_cluster_losses = cluster_losses
		centres = deepcopy(centres_next)

"""
DataTracker class wraps tracking the loss and number of training generation histories.
"""
class DataTracker:
	def __init__(self) -> None:
		self.loss_hist = []
		self.n_gen_hist = []

	def update(self, losses):
		self.loss_hist.append(losses[-1])
		self.n_gen_hist.append(len(losses))

"""
KineticOptimizerDataTracker class extends the DataTracker class and wraps tracking the counter of instances where a kinetic
optimizer outperformed or matched the performance of the k-means clustering algorithm.
"""
class KineticOptimizerDataTracker(DataTracker):
	def __init__(self) -> None:
		super().__init__()
		self.counter_ko_lte = 0
		self.counter_ko_lt = 0
	
	def update(self, losses_ko, losses_kmeans):
		super().update(losses=losses_ko)
		self.counter_ko_lte += losses_ko[-1] <= losses_kmeans[-1]
		self.counter_ko_lt += losses_ko[-1] < losses_kmeans[-1]

"""
DataSummary wraps the utility of reducing a DataTracker object for output.
"""
class DataSummary:
	def __init__(self, tracker) -> None:
		assert isinstance(tracker, DataTracker)
		self.loss = np.average(tracker.loss_hist)
		self.n_gens = np.average(tracker.n_gen_hist)

"""
KineticOptimizerDataSummary wraps the utility of reducing a KineticOptimizerDataTracker object for output.
"""
class KineticOptimizerDataSummary(DataSummary):
	def __init__(self, tracker, n_trials) -> None:
		super().__init__(tracker=tracker)
		assert isinstance(tracker, KineticOptimizerDataTracker)
		self.p_ko_lte = tracker.counter_ko_lte / n_trials
		self.p_ko_lt = tracker.counter_ko_lt / n_trials

"""
Runs the repeatable experiments to collect and aggregate the various implemented algorithms.
Returns a tuple containing the arbitrarily arranged DataSummary and KineticOptimizerDataSummary results.
"""
def run_data_collection(k, c, σ_classes, σ_samples, d, n_samples, n_trials, min_dist, min_loss) -> tuple:
	# Need to add a tracker for each type of experiment being ran.
	kmeans_tracker = DataTracker()
	iko_tracker = KineticOptimizerDataTracker()
	nko_tracker = KineticOptimizerDataTracker()

	for trial_count in range(1, n_trials + 1):
		items = []
		for _ in range(c):
			items.extend(np.random.normal(loc=np.random.normal(scale=σ_classes, size=d), scale=σ_samples, size=(n_samples,d)))

		init_indices = np.random.choice(len(items), size=k, replace=False)

		losses_kmeans = run_kmeans(init_indices=init_indices, items=items, k=k)
		kmeans_tracker.update(losses=losses_kmeans)
		iko_tracker.update(losses_ko=run_iko_loss_timeout(init_indices=init_indices, items=items, k=k, min_loss=min_loss, f_Δ=f_Δ), losses_kmeans=losses_kmeans)
		nko_tracker.update(losses_ko=run_nko_loss_timeout(init_indices=init_indices, items=items, k=k, min_loss=min_loss, f_Δ=f_Δ, α=np.min(items), β=np.max(items)), losses_kmeans=losses_kmeans)
	
	return (
			DataSummary(kmeans_tracker, n_trials=n_trials),
			KineticOptimizerDataSummary(iko_tracker, n_trials=n_trials),
			KineticOptimizerDataSummary(nko_tracker, n_trials=n_trials)
		)

"""
Multiprocessing target function for running the tests, then coordinating when the output file is unlocked, exporting the
experiment metrics, and finally indicating back to the main thread that a thread was just freed as the job is being terminated.
"""
def run_collect_and_export_data(data_fp, k, c, σ_classes, σ_samples, d, n_samples, n_trials, min_dist, min_loss, lock, c_processes):
	kmeans_ds,\
	iko_tracker_ds,\
	nko_tracker_ds = run_data_collection(
		k=k, c=c, σ_classes=σ_classes, σ_samples=σ_samples, d=d, n_samples=n_samples,
		n_trials=n_trials, min_dist=min_dist, min_loss=min_loss
	)
	assert isinstance(kmeans_ds, DataSummary)
	assert isinstance(iko_tracker_ds, KineticOptimizerDataSummary)
	assert isinstance(nko_tracker_ds, KineticOptimizerDataSummary)
	row_data = ",".join(["{}"] * 16).format(
			k, c , σ_classes, σ_samples, d, n_samples,
			kmeans_ds.loss, kmeans_ds.n_gens,
			iko_tracker_ds.loss, iko_tracker_ds.n_gens, iko_tracker_ds.p_ko_lte, iko_tracker_ds.p_ko_lt,
			nko_tracker_ds.loss, nko_tracker_ds.n_gens, nko_tracker_ds.p_ko_lte, nko_tracker_ds.p_ko_lt
		) + '\n'
	if lock:
		with lock:
			with open(data_fp, mode='a') as ofs:
				ofs.write(row_data)
	else:
		with open(data_fp, mode='a') as ofs:
				ofs.write(row_data)
	if c_processes:
		c_processes.value -= 1

"""
Calculates the minimum number of samples required such that, for the given number of classes c and clusters k, there exists at
least one item for each cluster so each cluster can be randomly initialized to a unique item.
"""
def get_n_samples_range(k, c, n_samples_max, step_size):
	if (step_size * c) > k:
		n_samples_min = step_size 
	else:
		n_samples_min = step_size * (int)(np.ceil((k + 1) / c))
	return range(n_samples_min, n_samples_max + step_size, step_size)

if __name__ == "__main__":
	import itertools
	import os
	import multiprocessing

	np.random.seed(0)
	min_update_dist = 0.00001
	loss_history_filter_threshold = min_update_dist

	n_trials = 30
	cpu_count = 15
	batch_outputs = True
	print("cpu_count={}".format(cpu_count))

	output_dir = "outputs-updated"
	data_fp = "{}/data.csv".format(output_dir)

	k_min, k_max = (2, 30)
	c_min, c_max = (2, 30)
	σ_classes_min, σ_classes_max = (2, 30)
	σ_samples_min, σ_samples_max = (2, 30)
	d_min, d_max = (2, 30)
	parameter_step_size = 4

	n_samples_max = 60
	n_samples_step_size = 5

	if batch_outputs:
		cpu_count = min(cpu_count, (int)(np.ceil(n_samples_max / n_samples_step_size)))

	experiments_generator = itertools.product(
		range(k_min, k_max + 1, parameter_step_size),
		range(c_min, c_max + 1, parameter_step_size),
		range(σ_classes_min, σ_classes_max + 1, parameter_step_size),
		range(σ_samples_min, σ_samples_max + 1, parameter_step_size),
		range(d_min, d_max + 1, parameter_step_size)
	)

	csv_header = "k,c,std_classes,std_samples,d,n_samples,\
loss_kmeans,n_gens_kmeans,\
loss_iko_loss,n_gens_iko,p_iko_lte,p_iko_lt_loss,\
loss_nko_loss,n_gens_nko,p_nko_lte,p_nko_lt_loss"

	try:
		os.makedirs(output_dir, exist_ok=False)
		with open(data_fp, mode='w') as ofs:
			ofs.write(csv_header + '\n')
	except:
		# Remove the last set of experiments wrt the number of samples being tested with.
		import pandas as pd
		if not os.path.exists(data_fp): 
			with open(data_fp, mode='w') as ofs:
				ofs.write(csv_header + '\n')
		else:
			data_df = pd.read_csv(data_fp, dtype={'k':int, 'c':int, 'std_classes':int, 'std_samples':int, 'd':int})
			if len(data_df) > 0:
				last_row = data_df.iloc[-1]
				data_df = data_df.query("k != {} | c != {} | std_classes != {} | std_samples != {} | d != {}".format(
					last_row.k, last_row.c, last_row.std_classes, last_row.std_samples, last_row.d
				))
				data_df.to_csv(data_fp, index=False)
				if len(data_df) > 0:
					last_row = data_df.iloc[-1]
					last_k = (int)(last_row.k)
					last_c = (int)(last_row.c)
					last_std_classes = (int)(last_row.std_classes)
					last_std_samples = (int)(last_row.std_samples)
					last_d = (int)(last_row.d)

					# Advance the experiments generator to the progressed point.
					for k, c, std_classes, std_samples, d in experiments_generator:
						if k == last_k and c == last_c and std_classes == last_std_classes and std_samples == last_std_samples and d == last_d:
							break
	
	if cpu_count == 1:
		for k, c, σ_classes, σ_samples, d in experiments_generator:
			print("\rk={}, c={}, σ_classes={}, σ_samples={}, d={}        ".format(k, c, σ_classes, σ_samples, d), end='')
			for n_samples in get_n_samples_range(k=k, c=c, n_samples_max=n_samples_max, step_size=n_samples_step_size):
				run_collect_and_export_data(data_fp, k, c, σ_classes, σ_samples, d, n_samples, n_trials, min_update_dist, loss_history_filter_threshold, None, None)
	else:
		with multiprocessing.Pool(cpu_count) as mp_pool:
			with multiprocessing.Manager() as mp_manager:
				lock = mp_manager.Lock()
				async_result = None
				if batch_outputs:
					for k, c, σ_classes, σ_samples, d in experiments_generator:
						print("\rk={}, c={}, σ_classes={}, σ_samples={}, d={}        ".format(k, c, σ_classes, σ_samples, d), end='')
						async_result = mp_pool.starmap_async(run_collect_and_export_data,[
							(data_fp, k, c, σ_classes, σ_samples, d, n_samples, n_trials, min_update_dist, loss_history_filter_threshold, lock, None) for n_samples in get_n_samples_range(k=k, c=c, n_samples_max=n_samples_max, step_size=n_samples_step_size)
						])
						async_result.get()
				else:
					import time
					c_processes = mp_manager.Value("c_processes", 0)
					for k, c, σ_classes, σ_samples, d in experiments_generator:
						print("\rk={}, c={}, σ_classes={}, σ_samples={}, d={}        ".format(k, c, σ_classes, σ_samples, d), end='')
						n_samples_list = get_n_samples_range(k=k, c=c, n_samples_max=n_samples_max, step_size=n_samples_step_size)
						c_processes.value += len(n_samples_list)
						async_result = mp_pool.starmap_async(run_collect_and_export_data,[
							(data_fp, k, c, σ_classes, σ_samples, d, n_samples, n_trials, min_update_dist, loss_history_filter_threshold, lock, c_processes) for n_samples in n_samples_list
						])
						time.sleep(0.05)
						while c_processes.value > cpu_count:
							time.sleep(0.1) # wait before checking to see if a thread was freed
					if async_result is not None:
						async_result.get()
	print("")