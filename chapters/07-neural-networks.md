# Neural Networks

 ## Introduction

 Many of the most impressive achievements in machine learning stem from the development of "artificial neural networks."
 The earliest ideas for building algorithms based on models of neurons go back to the 40's, and peaked in the late 1950's and
 early 1960's with the development of the "perceptron", which was an early example of what we now call a multi-layer neural net.

 Hardware limitations, as well as a tendency for people to overstate the power of these early techniques, led to the abandonment
 of these ideas for nearly 50 years until Geoffrey Hinton and others returned to them with the benefit of the dramatic improvements
 in computer power, specialized hardware such as GPUs, and a number of crucial improvements to algorithms for optimization.  Since then neural networks have shown an amazing
 ability to "learn" and have overcome challenges in image recognition and other classic problems in artificial intelligence,
 culminating in the invention of the attention mechanism, the transformer, and the LLM.

 ## Basics of Neural Networks

 At its heart, a neural network is a function $F$ that is built out of two simple components:
    
    - linear maps (matrices)
    - simple non-linear maps (activation functions).

The function $F$ is the composition of these types of functions so that $F(x) = \cdots \sigma_2\circ M_2\circ\sigma_1\circ M_1$.

The underlying idea for building a neural network to solve a particular problem is to construct the function $F$ with the matrices having
random entries to start, then taking a large set of data $(x_i,y_i)$ where the goal is to adjust $F$ so that $F(x_i)$ is as close to $y_i$
as possible (this is called "training").

So for example if the goal is to build a neural network that recognizes pictures of cats, one starts with a big library of images (a training set) $x_i$ and labels $y_i$
"cat" (or 1) and "not-cat" (or zero).  Then one tries to build a function $F$ that attaches a probability to an image between $0$ and $1$ measuring
how sure the function is that the picture is, or is not, a cat.  To do this, one starts with an $F$ with random initial parameters (weights) in the matrices making up $F$,
and then computes $F(x_i)$ and compares it to $y_i$.  Using an optimization algorithm one adjusts the weights until $F(x_i)$ is close to $1$ when $y_i$ is one,
and close to zero when $y_i$ is zero.  Eventually one gets a function which, hopefully, can recognize images that were not in its training set and attach high probabilities
to pictures of cats and low probabilities to pictures of not-cats.

In fact, both linear and logistic regression fit into this framework and are very simple examples of neural networks.  In the case of linear regression,
we have "training data" $(x_i,y_i)$ and our goal is to find a matrix $M$ so that the function $Y=MX$ is a good approximation to a function giving $y_i=Mx_i$;
the error is measured by the mean-squared error and we can find $M$ either analytically or by gradient descent.

