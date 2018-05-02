import pickle
import numpy as np
import matplotlib.pyplot as plt
from plot_NELperformance import calculate_mean_rew_per_time

##
filenames = ['reward_1', 'loss_1', 'epsilon_1', 'trigger_1']


def plot_with_triggers(xvalues,yvalues,triggers):
    plt.plot(xvalues,yvalues,'b',linewidth=0.5) #same
    plt.plot(yvalues,'b',linewidth=1.0) #same

    for trigger in triggers:
        plt.axvline(x=trigger,color='r',linewidth=1.5)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.show()

def plot_multiple_subplots_with_triggers(data,triggers,labels):
    number_of_plots=len(data);
    plt.figure()

    for i in range(number_of_plots):
        plt.subplot(number_of_plots,1,i+1)
        plt.plot(data[i][0],data[i][1],'b',linewidth=0.5)
        plt.xlabel(labels[i][0])
        plt.ylabel(labels[i][1])
        if i!=2:
            for trigger in triggers:
                plt.axvline(x=trigger,color='r',linewidth=1.0)
        else:
            for trigger in triggers:
                plt.axvline(x=trigger/2,color='r',linewidth=1.0)

    plt.show()

if __name__ == "__main__":
    # with open('rews_baseline_0.pkl', 'rb') as f:
    #      data = np.asarray(pickle.load(f))
    # data=np.random.randint(1,high=100,size=500)
    # steps=np.arange(len(data))
    # triggers=[20,100,151,400,330]
    # triggers=[20123,103150,151351,403150,33350]

    # plot_with_triggers(steps,data,triggers)
    # data2=[]
    # labels=[]

    # for i in [1,2,3]:
    #     yvalues=np.random.randint(1,high=100,size=500)
    #     xvalues=np.arange(len(yvalues))
    #     labels.append(['xlabel','ylabel'])
    #     data2.append([xvalues,yvalues])
    # plot_multiple_subplots_with_triggers(data2,triggers,labels)
    # plt.plot(xvalues,mean_rewards)
    # plt.show()
    data3=[]
    labels3=[]
    with open(filenames[0], 'r') as f:
        data = f.read()
    res = [float(i) for i in data.split()]
    yvalues=res
    xvalues=np.arange(len(yvalues))
    # plt.plot(xvalues,yvalues)
    # plt.show()
    data3.append([xvalues,yvalues])
    labels3.append(['steps', 'reward'])

    mean_rewards=calculate_mean_rew_per_time(yvalues)
    data3.append([xvalues,mean_rewards])
    labels3.append(['steps', 'mean cumulative reward'])



    with open(filenames[1], 'r') as f:
        data = f.read()
    res = [float(i) for i in data.split()]
    yvalues = res
    xvalues = np.arange(len(yvalues))
    # plt.plot(xvalues, yvalues)
    # plt.show()
    mean_rewards = calculate_mean_rew_per_time(yvalues)
    data3.append([xvalues, yvalues])
    labels3.append(['steps (loss)', 'loss'])

    with open(filenames[2], 'r') as f:
        data = f.read()
    res = [float(i) for i in data.split()]

    yvalues = res
    xvalues = np.arange(len(yvalues))
    # plt.plot(xvalues, yvalues)
    # plt.show()
    data3.append([xvalues, yvalues])
    labels3.append(['steps', 'epsilon'])


    with open(filenames[3], 'r') as f:
        data = f.read()
    res = [float(i) for i in data.split()]
    triggers = res



    plot_multiple_subplots_with_triggers(data3,triggers,labels3)
    print('finished')




