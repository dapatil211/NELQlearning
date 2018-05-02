# import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_with_triggers(xvalues,yvalues,triggers):
    plt.plot(xvalues,yvalues,'b',linewidth=1.0) #same
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
        plt.plot(data[i][0],data[i][1],'b',linewidth=1.0)
        plt.xlabel(labels[i][0])
        plt.ylabel(labels[i][1])

        for trigger in triggers:
            plt.axvline(x=trigger,color='r',linewidth=1.5)
    plt.show()

if __name__ == "__main__":
    # with open('rews_baseline_0.pkl', 'rb') as f:
    #     data = np.asarray(pickle.load(f))
    data=np.random.randint(1,high=100,size=500)
    steps=np.arange(len(data))
    triggers=[20,100,151,400,330]
    plot_with_triggers(steps,data,triggers)
    data2=[]
    labels=[]
    for i in [1,2,3]:
        yvalues=np.random.randint(1,high=100,size=500)
        xvalues=np.arange(len(yvalues))
        labels.append(['xlabel','ylabel'])
        data2.append([xvalues,yvalues])
    plot_multiple_subplots_with_triggers(data2,triggers,labels)
