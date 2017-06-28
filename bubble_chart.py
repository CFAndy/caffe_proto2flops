import matplotlib.pyplot as plt
import collections
import numpy as np
import sys

def main():

    file_name = sys.argv[1]
    file = open(file_name, "r")
    lines = file.readlines()
    name = []
    mad = []
    weight = []
    accuracy = []
    for x in range(1,len(lines)):
        cols = lines[x].split("\t")
        print(cols)
        name.append(cols[0][:-9])
        mad.append(float(cols[1]))
        weight.append(float(cols[3]))
        accuracy.append(float(cols[2]))
    x = np.array(mad)
    y = np.array(accuracy)
    w = np.array(weight)*100
    for i in range(0, x.size):
        plt.scatter(x[i], y[i], s=w[i], marker='o', label=name[i])
    # Chart title
    plt.title('Topology weight vs flops vs accuracy ')

    # y label
    plt.ylabel('Top-1 accuracy')

    # x label
    plt.xlabel('Giga_MAD')

    #plt.clabel('Milli_params')

    # and a legend
    plt.legend(scatterpoints=1, loc='best',   ncol=1,   markerscale=0.1,   fontsize=12)
    plt.show()

if __name__ == '__main__':
    main()