---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Effect of Initialization on Optimization Trajectory in 2D"
subtitle: "Visualizing how Initialization effects optimization"
summary: "The purpose of this post is to demonstrate the importance of Initialization"
authors: [Pranay Modukuru]
tags: [Deep Learning, Optimization, First post]
categories: []
date: 2020-03-05T01:53:52+01:00
lastmod: 2020-03-05T01:53:52+01:00
featured: false
draft: false


# Optional external URL for project (replaces project detail page).
external_link: "https://medium.com/datadriveninvestor/effect-of-initialization-on-optimization-trajectory-129746a2bb9d?source=friends_link&sk=918a3508f94da3e3da25b5b2eaf52356"

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

[View post](https://medium.com/datadriveninvestor/effect-of-initialization-on-optimization-trajectory-129746a2bb9d?source=friends_link&sk=918a3508f94da3e3da25b5b2eaf52356) on Medium.


<!-- ### The Optimization Problem
In simple words, finding a minimum value for a given equation is considered as optimization. This has many applications in real life - finding the fastest path when traveling from one place to other, job shop scheduling, air traffic management etc,. The optimization has been the back bone of machine learning, where the algorithms are expected to extract knowledge from huge volumes of data.

Optimization plays a major role Neural Networks where there are millions of parameters and the goal is to find the right set of parameters to correctly represent the data. There has been a lot of research in this field and many algorithms have been developed for effective optimization. Even though the performance of the optimizer has improved a lot, there is another problem that the optimization depends upon i.e. the initial point. The trajectory of optimization is largely dependant on the initialisation. This has been studied and numerous initialization techniques have been proposed to effectively exploit the power of optimization algorithms.

In this post we are going to see how the initialization can affect the performance of some of the optimization algorithms until day. Although, we are using a two dimensional problem here since it is easy to visualize, the initialization problem becomes more prevalent when there are millions of parameters (Neural Networks).


### The Task
Initialize x, y and use gradient descent algorithms to find the optimal values of x and y such that the value of the Beale function is zero (or as low as possible).


### A brief introduction to optimization algorithms
We are going to consider three popular optimization algorithms, since we are more concerned about the initialization these will be sufficient for our analysis.
1. **Stochastic Gradient Descent** - The stochastic gradient descent (SGD) algorithm performs one update at a time computing gradients at each step.
2. **Momentum** - Overcomes the difficulty of slow updates of stochastic gradient descent by considering the momentum of gradients over a period of time.
3. **Adam** - Considered to be the most popular optimization algorithm. It takes into consideration the first and second moments i.e. the exponentially decaying average of past gradients and squared gradients.

A more detailed explanation about gradient descent optimization algorithms, please read this [post](https://ruder.io/optimizing-gradient-descent/) by [Sebastian Ruder](https://ruder.io/).

###### Importing required libraries
We are going to use the autograd functionality of [PyTorch](https://pytorch.org/) for getting the gradients and matplotlib for plotting the trajectories.

```Python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

import warnings
warnings.filterwarnings("ignore")
```

### The Beale Function
$$
f(x, y) =  (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 +(2.625 - x + xy^3)^2\tag{1}
$$

* Beale function is a multimodal non-convex continuous function defined in two dimensions.
* It is usually evaluated in the range $(x, y) \in [-4.5, 4.5]$.
* The function has only one global minimum at $(x, y) = (3, 0.5)$.

The [Beale](https://www.sfu.ca/~ssurjano/beale.html) Function, a two dimentional function is chosen to make visualizations simple.
###### Visualizing the Beale Function
As the Beale function is a two variable function ranging between -4.5 and 4.5, we can generate a meshgrid using numpy to pass all the possible values of x and y to the function. This enables us to have the output of the beale's function at each possible point, we can use these outputs to visualize the function in a contour plot.

As we are relating the optimization problem with neural networks, we will refer to **(x, y)** as **(w1 , w2)**. Also, when using a neural network we refer to objective function as a loss function and the output of the function as loss. In this case, we refer to the Beale's function as **loss function** and the outputs as **losses**


```python
# Defining function
f  = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

# Defining the range of w1 and w2, step size
w1_min, w1_max, w1_step = -4.5, 4.5, .2
w2_min, w2_max, w2_step = -4.5, 4.5, .2

# Global minima of the function
minima_ = [3, 0.5]

# generating meshgrid
w1, w2 = np.meshgrid(np.arange(w1_min, w1_max + w1_step, w1_step),
                     np.arange(w2_min, w2_max + w2_step, w2_step))
losses = f(w1, w2)
```
We will now plot the losses on a contour plot with the following code.

```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.contour(w1, w2, losses, levels=np.logspace(0, 5, 35),
                    norm=LogNorm(), cmap=plt.cm.jet, alpha = 0.8)
ax.plot(*minima_, 'r*', color='r',
                    markersize=10, alpha=0.7, label='minima')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_xlim((w1_min, w1_max))
ax.set_ylim((w2_min, w2_max))
ax.legend(bbox_to_anchor=(1.2, 1.))
ax.set_title("Beale Function")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
```

###### Output
As seen in the image, blue region indicates lower values and the red region is higher values of the Beale's function. The minima (3, 0.5) is indicated with a star.

{{< figure library="1" src="optimization-trajectory-visualization/beale_func.png" >}}

###### Setting up Parameters
As we are using PyTorch, we need to have parameters that are to be optimized put into a ```nn.Module``` class. The ```__init__()``` takes (x, y) as inputs to initialize the parameters (w1, w2).  Also, we are going to write the beale's equation in the forward function.


```python
class Net_Beale(torch.nn.Module):
    def __init__(self, x, y):
        super(Net_Beale, self).__init__()
        self.w1 = torch.nn.Parameter(torch.tensor([x]))
        self.w2 = torch.nn.Parameter(torch.tensor([y]))

    def forward(self):
        # Beale Function Equation
        a = (1.5 - self.w1 + self.w1*self.w2)**2
        b = (2.25 - self.w1 + self.w1*self.w2**2)**2
        c = (2.625 - self.w1 + self.w1*self.w2**3)**2
        return a+b+c
```
###### Optimizing and Saving Trajectory
Since we are interested in tracking the path of the optimization, we need to collect the parameters at each step/desired steps and save them for plotting.

The below function initialises the parameters of the network, initializes an optimizer and runs the optimization for the specified number of steps while collecting the path of the parameters.


```python
def get_trajectory(x, y, optim, lr, epochs, interval=1):

    # Initialize Network
    net = Net_Beale(x,y)

    # Initialize Optimizer
    if optim == "sgd":
        optim = torch.optim.SGD(net.parameters(), lr)
    elif optim == "mom":
        optim = torch.optim.SGD(net.parameters(), lr, momentum=0.9)
    elif optim == "adam":
        optim = torch.optim.Adam(net.parameters(), lr)

    # Initialize Trackers
    w_1s = []
    w_2s = []

    # Run Optimization
    for i in range(epochs):
        optim.zero_grad()
        o = net()
        o.backward()

        if i % interval == 0:
            # Append current w1 and w2 to trackers
            w_1s.append(net.w1.item())
            w_2s.append(net.w2.item())
        optim.step()

    w_1s.append(net.w1.item())
    w_2s.append(net.w2.item())

    # Join w1's and w2's into one array
    trajectory = np.array([w_1s, w_2s])   

    return trajectory
```

###### Comparison between trajectories
After collecting the paths of parameters with different algorithms, we are going to plot them on the Beale Function Contour plot. The below function takes in the initial position, list of optimizers and corresponding learning rates and epochs and plots the trajectories of algorithms with specified settings.  

```python
def compare_trajectories(x, y, epochs, optims, lrs):

    colors = ['k', 'g', 'b', 'r', 'y', 'c', 'm']
    trajectories = []
    names = []
    # Loop on all optimizers in list
    for ep, optim, lr in zip(epochs, optims, lrs):
        trajectory = get_trajectory(float(x), float(y), optim=optim, lr=lr, epochs=ep)
        names.append(optim)
        trajectories.append(trajectory)

    # Plot the Contour plot of Beale Function and trajectories of optimizers
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(w1, w2, losses, levels=np.logspace(0, 5, 35),
                        norm=LogNorm(), cmap=plt.cm.jet, alpha = 0.5)

    for i, trajectory in enumerate(trajectories):
        ax.quiver(trajectory[0,:-1], trajectory[1,:-1], trajectory[0,1:]-trajectory[0,:-1],
                  trajectory[1,1:]-trajectory[1,:-1], scale_units='xy', angles='xy', scale=1,
                  color=colors[i], label=names[i], alpha=0.8)

    start_ =[x,y]
    ax.plot(*start_, 'r*', color='k',markersize=10, alpha=0.7, label='start')
    ax.plot(*minima_, 'r*', color='r',markersize=10, alpha=0.7, label='minima')
    ax.set_xlabel('w1')
    ax.set_ylabel('w2')
    ax.set_xlim((w1_min, w1_max))
    ax.set_ylim((w2_min, w2_max))
    ax.set_title("Initial point - ({},{})".format(x,y))
    ax.legend(bbox_to_anchor=(1.2, 1.))
    fig.suptitle("Optimization Trajectory")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
```

### Trying Different Initial Points

After setting up everything, we are now ready to compare the three algorithms with different initial points.

The learning rates observed to be working in this problem
* SGD      - 0.0001
* momentum - 0.0001
* Adam     - 0.01

We are going to use the same learning rate for different initial points considered for respective algorithms to keep the analysis simple and since we are not doing hyperparameter tuning. Feel free to download this [notebook](https://github.com/pranaymodukuru/OptimizationTrajectoryPlots/blob/master/OptimizationTrajectory_Initialization.ipynb) and trying out different hyperparameters and initial points.

```python
# Settings for optimizers
epochs = [10000] * 3
optims = ['sgd', 'mom', 'adam']
lrs = [0.0001, 0.0001, 0.01]
```
###### Case 1 : A point close to minima
```python
#  A point closer to minima
x = 2.5
y = 2.
compare_trajectories(x, y, epochs, optims, lrs)
```

{{< figure library="1" src="optimization-trajectory-visualization/close_to_minima.png" >}}

All the three reach the global minima, lets move a little further and see what happens.

###### Case 2 : Moving a little further
```python
# A little away in the same region
x = 1.5
y = 2.5
compare_trajectories(x, y, epochs, optims, lrs)
```
{{< figure library="1" src="optimization-trajectory-visualization/a_little_further.png" >}}

Moving a little further from where we have started has made a huge difference on how an optimizer moves the parameters towards minimum. As seen in the figure above **adam** optimizer moves towards a local minimum and is stuck there, whereas **sgd** and **momentum** reach the global minimum. Things to notice, we are not changing the learning rate here, as we are focusing on effect of initialization on optimization.

###### Case 3 : Moving a little further
```python
# Lower left region
x = -4
y = -4
compare_trajectories(x, y, epochs, optims, lrs)
```
{{< figure library="1" src="optimization-trajectory-visualization/lower_left.png" >}}

(more example can be found in this [notebook](https://github.com/pranaymodukuru/OptimizationTrajectoryPlots/blob/master/OptimizationTrajectory_Initialization.ipynb))

### Conclusion
The initial point plays a crucial role in optimization problems. Here we are trying to solve a two dimensional problem which is fairly easy when compared to finding a minima when we have a large dataset and more than million parameters (dimensions).

Although we are not tuning the hyperparameters here, we can effectively drive the optimization in right direction with the right set of hyperparameters.

## References

1. https://mitpress.mit.edu/books/optimization-machine-learning
2. https://en.wikipedia.org/wiki/Test_functions_for_optimization
3. http://benchmarkfcns.xyz/benchmarkfcns/bealefcn.html
4. https://ruder.io/optimizing-gradient-descent/
5. http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
6. https://communities.sas.com/t5/SAS-Communities-Library/Mathematical-Optimization-in-our-Daily-Lives/ta-p/504724# -->
