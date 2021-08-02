# %%
# -*- coding: utf-8 -*-
#  File: plot_db.py
#  Project: 'Plot_decision_boundary'
#  Created by Gennady Matveev (gm@og.ly) on 02-08-2021.
#  Copyright 2021. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = 10, 6
plt.style.use('ggplot')


# Function to plot model decision boundary:
def plot_db(X, y, model, features=(0, 1), proba=False, grid_size=100):
    '''
    Plots the model's decision boundaries or color-coded
    probabilities for class 0 in classification tasks.

    Arguments:

    X:          array of features
    y:          corresponding array of targets
    features:   tuple of two selected features
    proba:      if True, plots probabilities for class 0,
                else plots decision boundary; bool, default - False
    grid_size:  defines grid size, very high values will slow 
                down plotting; int, default = 100
    '''
    # take the selected pair of features from X
    X = X[:, list(features)]

    # define bounds of the domain
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

    # define the x and y scale

    x1grid = np.linspace(min1, max1, grid_size)  # change to linspace
    x2grid = np.linspace(min2, max2, grid_size)

    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))

    # fit the model
    model.fit(X, y)

    # choose between class prediction and probability
    if proba:
        # make predictions for the grid
        yhat = model.predict_proba(grid)
        # keep just the probabilities for class 0
        yhat = yhat[:, 0]
    else:
        yhat = model.predict(grid)

    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)

    # plot the grid of x, y and z values as a surface

    c = plt.contourf(xx, yy, zz, cmap='RdBu')#'summer'
    # add a legend, called a color bar

    plt.colorbar(c)
    # create scatter plot for samples from each class

    num_classes = len(np.unique(y))
    for class_value in range(num_classes):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    # data =  np.loadtxt(fname = './data/scatter01.csv', delimiter = ',')
    # X, y = data[:,:-1], data[:,-1]

    model = LogisticRegression()

    plot_db(X, y, model, features=(2, 3), proba=False)


if __name__ == '__main__':
    main()


#%%
