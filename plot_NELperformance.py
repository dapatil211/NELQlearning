import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from six.moves import cPickle
from collections import deque
import numpy as np

def read_and_calculate_score(f_path, ind):

	data = cPickle.load( open( f_path, "rb" ) )
	rews = data[ind]

	rews = get_rewards_per_step(rews)

	print('tot_rew :', np.sum(rews))
	print('ave_rew :', np.mean(rews))

	return calculate_mean_rew_per_time(rews)


# computes total_rewards/number_of_steps per step
def calculate_mean_rew_per_time(rewards):
	mean_rews = []
	tot_rews = 0
	for i in range(len(rewards)):
		tot_rews += rewards[i]
		mean_rews.append(tot_rews/(i+1.0))
	return mean_rews


# back calculates the rewards per step from the rewards in the last 100 steps
def get_rewards_per_step(data):

	rews = [0]
	for i in range(len(data)):
		if i < 100 and i > 0:
			rews.append(data[i] - data[i-1])
		if i >= 100:
			deleted_val = rews[i-100]
			curr_val = data[i]
			prev_val = data[i-1]
			if deleted_val == 0:
				if curr_val - prev_val == 0:
					rews.append(0)
				else:
					rews.append(1)
			else:
				if curr_val - prev_val == 0:
					rews.append(1)
				else:
					rews.append(0)

	return rews


def main():

	running_mean_rew_qnet_20m = []
	running_mean_rew_qnet = []
	running_mean_agent = []

	# load baseline 20m run running average rewards
	for j in range(1):
	
		file_dir = "baseline_20m/"
		f_name = "rews_baseline_20m.pkl"
		
		running_mean_rew_qnet_20m = cPickle.load( open( file_dir+f_name, "rb" ) )

	# load baseline 500k run running average rewards
	for j in range(10):
	
		file_dir = "baseline_500k/"
		f_name = "rews_baseline_" + str(j) + ".pkl"
		
		running_mean_rew_qnet.append(cPickle.load( open( file_dir+f_name, "rb" ) ))


	file_name = "train_stats.pkl"
	# load data from the agent's file 
	# the pickle file consists of total rewards obtained in the previous 100 steps
	for j in range(10):

		file_dir = "Results/n=100 steps/"
		file_dir2 = "outputs_" + str(j)
		
		file_dir = file_dir + file_dir2 + "/"
		ind = 1
		
		running_mean_agent.append(read_and_calculate_score(file_dir+file_name, ind))


	mean_qnet = np.mean(running_mean_rew_qnet,axis=0)
	std_qnet = np.std(running_mean_rew_qnet,axis=0)

	mean_agent = np.mean(running_mean_agent,axis=0)
	std_agent = np.std(running_mean_agent,axis=0)



	fig, ax = plt.subplots()
	
	# create x-axis data
	x = range(1,len(mean_qnet)+1)
	print(len(x))
	x2 = range(1,len(mean_agent)+1)
	print(len(x2))

	# plot 20m run performance data
	ax.plot(running_mean_rew_qnet_20m,linewidth=5, color='black', label='Q-net Baseline')

	# plot 500k run performance data
	ax.plot(mean_qnet,linewidth=5, color='red', label='Q-net Baseline')
	ax.fill_between(x, mean_qnet - std_qnet, mean_qnet + std_qnet,color='red',alpha=0.4)

	# plot agent performance data
	ax.plot(mean_agent,linewidth=5, color='green', label='Agent')
	ax.fill_between(x2, mean_agent - std_agent, mean_agent + std_agent,color='green',alpha=0.4)


	ax.set_ylim([0,0.1])
	ax.set_title('Baseline Q-Learning Results')
	ax.set(xlabel='Steps in the environment', ylabel='R')
	ax.legend(loc='upper right', fontsize='x-large', ncol=2)
	ax.grid()

	plt.plot()

	fig.savefig("baselines_results.png")
	plt.show()


if __name__ == '__main__':
    main()
