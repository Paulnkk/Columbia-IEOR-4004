# Solving-L2-Logistic-Regression-Problem

In the course "Machine Learning and Optimization" I implemented a trust-region method which solves the L2 regularized Logistic Regression Problem. The Dogleg method is used to solve the subproblems. The target parameter, beta^(star), is calculated using a Maximum Likelihood approach.

I have implemented the objective function as well as its gradient and Hessian matrix. In addition, I used the following parameters to initialize the method: 
lambda = 10^(-5), beta^0 = 0 (starting point), t^0 = 1 (starting radius), t^(hat) = 10^4 (maximum radius), Eta = 0.05 and termination tolerance = 10^(-5).
