# plot_pd
 
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
