import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scatterplot(X, Y, xLabel, yLabel, xticks=None, yticks=None, colors=None):

    plt.scatter(X, Y, s=20, c=colors)
    plt.xlabel(xLabel)
    if xticks is not None:
        plt.xticks(np.arange(xticks.size), xticks)
    if yticks is not None:
        plt.yticks(np.arange(yticks.size), yticks)
    plt.ylabel(yLabel)


def a():
    df = pd.read_csv("./3.csv")
    scatterplot(df["X1"],df["X2"],xLabel="X1",yLabel="X2", colors=df["Y"])
    plt.savefig("./plots/9.7.3.a.png")

def b():
    df = pd.read_csv("./3.csv")
    scatterplot(df["X1"], df["X2"], xLabel="X1", yLabel="X2", colors=df["Y"])
    x1,x2 = [4.0,2.0],[3.5,1.5]
    plt.plot(x1,x2,marker="o")

    plt.savefig("./plots/9.7.3.b.png")
if __name__ == '__main__':
    #a()
    b()