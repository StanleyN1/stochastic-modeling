# stochastic-modeling

Abstract: Inferring dynamic parameters of stochastic differential equations from data is a fundamental problem in data science. We compare estimated parameters obtained from **temporal normalizing flows** and analytic formulas such as from the **Kramers Moyal Formula** and the associated **Fokker-Planck solution**.

Implementing Euler's method to Stochastic Differential Equations. Applying derivations from Kramers Moyal Formula as presented by Dai et al.

In short, given simulated data in first graph, approximate the diffusion and drift functions in the second graph. The third graph highlights an approximation of the drift functions.

![alt text](https://github.com/StanleyN1/stochastic-modeling/blob/main/pics/actualfunc.png?raw=True)

Figure 1: goal function to approximate (diffusion is only shown)

![alt text](https://github.com/StanleyN1/stochastic-modeling/blob/main/pics/simulated.png)

Figure 2: data generated by drift and diffusion over time.

![alt text](https://github.com/StanleyN1/stochastic-modeling/blob/main/pics/approxpoly.png)

Figure 3: approximation and polynomial interpolation of drift and diffusion using eqs (2) and (3) by Dai et al.

References:

  Cheng et al. https://www.sciencedirect.com/science/article/abs/pii/S0378437119310325

  Dai et al. https://arxiv.org/abs/2001.01412

  https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method
