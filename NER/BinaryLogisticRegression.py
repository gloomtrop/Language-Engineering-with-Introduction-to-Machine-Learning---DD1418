from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.01  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 100 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)
    LAMBDA = 0.001

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)



    # ----------------------------------------------------------------------


    def loss(self, x, y):
        """
        Computes the loss function given the input features x and labels y
        
        :param      x:    The input features
        :param      y:    The correct labels
        """
       
        sig = self.sigmoid(np.dot(self.theta,np.transpose(self.x)))
        
        return np.sum(-y*np.log(sig)-(1-y)*np.log(1-sig))/len(y) + self.LAMBDA*np.dot(self.theta, self.theta)/(2*len(y))


    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + np.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """

        # REPLACE THE COMMAND BELOW WITH YOUR CODE
        sig = self.sigmoid(np.dot(self.theta,self.x[datapoint]))

        return sig**label*(1-sig)**(1-label)

    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        # YOUR CODE HERE
        

    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        
        # YOUR CODE HERE


    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        # YOUR CODE HERE


    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)
        self.LEARNING_RATE = 0.1
        # YOUR CODE HERE
        t = 0
        start = time.time()
        conv = 1
        
        while conv>0 and t < self.MAX_ITERATIONS*len(self.y):
            randi = random.randint(0,len(self.y)-1)
            self.gradient = self.x[randi,:]*(self.sigmoid(np.dot(self.theta,self.x[randi,:])) - self.y[randi]) + self.LAMBDA*self.theta/len(self.y)
            self.theta = self.theta - self.LEARNING_RATE*self.gradient

            t += 1
            if t %100 == 0:
                self.update_plot(self.loss(self.x, self.y))
            conv = np.dot(self.gradient,self.gradient)-self.CONVERGENCE_MARGIN
        print("Iterations",t)
        print("Time",time.time()-start, "s")


    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        # YOUR CODE HERE
        self.LEARNING_RATE = 0.1
        t = 0
        size = self.MINIBATCH_SIZE
        start = time.time() 
        while np.dot(self.gradient,self.gradient)>self.CONVERGENCE_MARGIN or t == 0:
            ri = random.randint(0, len(self.y)-1-size)

            self.gradient = np.dot(np.transpose(self.x[ri:ri + size,:]),
            (self.sigmoid(np.dot(self.theta,np.transpose(self.x[ri:ri + size,:]))) - self.y[ri:ri + size]))/size + self.LAMBDA*self.theta/len(self.y)

            self.theta = self.theta - self.LEARNING_RATE*self.gradient

            if t %50 == 0:
                self.update_plot(self.loss(self.x, self.y))
            t += 1
        print("Iterations",t)
        print("Time",time.time()-start, "s")


    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)
        # YOUR CODE HERE
        
        t = 0
        self.LEARNING_RATE = 0.1
        start = time.time()
        while np.dot(self.gradient,self.gradient)>self.CONVERGENCE_MARGIN or t == 0:
            self.gradient = np.dot(np.transpose(self.x),(self.sigmoid(np.dot(self.theta,np.transpose(self.x))) - self.y))/len(self.y) + self.LAMBDA*self.theta/len(self.y)
            self.theta = self.theta - self.LEARNING_RATE*self.gradient
            if t %10 == 0:
                self.update_plot(self.loss(self.x, self.y))
            t += 1
        print("Iterations",t)
        print("Time",time.time()-start, "s")



    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:')

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))

        N = np.sum(confusion)
        print("Accurracy:", (confusion[0,0]+confusion[1,1])/N ,"\n-----------------------\n")
        print("Name: ")
        print("Precision:",(confusion[1,1]/(confusion[1,1]+confusion[1,0])))
        print("Recall:",(confusion[1,1]/(confusion[1,1]+confusion[0,1])))
        print("\n-----------------------\n" + "No Name:")
        print("Precision:", (confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])))
        print("Recall:", (confusion[0, 0] / (confusion[0, 0] + confusion[1, 0])))
        

    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()
