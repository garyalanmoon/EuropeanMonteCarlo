This program will price European options, under the standard Black-Scholes assumptions, using Monte Carlo simulations. The primary needed market/option parameters are the following:
S_0 = current price of the underlying asset (stock)
r = risk-free interest rate (on annualized basis)
sig = market volatility (e.g., standard deviation of market returns)
K = strike price
T = expiry (how long until option expires)

We also need an option-type parameter. The functions take in 'call' or 'put', but this is converted to a parameter q as
'call' -> q = 1
'put' -> q = -1.

Finally, we need a parameter N that determines the number of simulations to run.

The program will first price the option using standard Monte Carlo simulation. Since, in a standard Black-Scholes world, the underlying stock will evolve according to a geometric Brownian motion with
constant drift and constant volatility. The corresponding SDE has a closed-form analytical solution which we exploit to avoid simulating the entire path of the option.
Our approach is to 
(1) simulate N time T stock prices S_T,
(2) compute the discounted (from time T to time 0) payoff of the option for each S_T,
(3) compute the expected value of the discounted payoffs.
This expected value gives the approximation of the option price. 

The program also computes a number of statistics to help understand the error involved in the pricing. 
(1) The Black-Scholes formula is used to price the option and the error is reported.
(2) The variance in the discounted payoffs is computed and reported.
(3) A 95% CI for the option price is constructed.
Note that (1) relies on having a closed-form analytical solution to the pricing equation which is often not the case. However, the other 2 can be computed more generally.

Next, the program reruns the Monte Carlo simulation, but using a variance reduction strategy. The primary benefit of this strategy is to reduce the error in the price for a fixed number of simulations N
by reducing the variance of the discounted payoffs. It can also result in a slight speedup by only requiring the generation of N/2 pseudorandom numbers, but this process is generally not the speed
bottleneck, so we do not expect a major improvement in speed.

The process for pricing the option is as follows:
(1) generate N/2 pseudorandom numbers Z,
(2) define the antithetic variats Z_anti = -Z,
(3) simulate N time T stock prices S_T (N/2 using Z and N/2 using Z_anti),
(4) compute expected value of discounted payoffs.
Again, the price is given by the above expected value.

The program again produces the above summary statistics to evaluate the effectiveness of the variance reduction strategy.

Lastly, the program runs a convergence study on both pricers. We expect that Monte Carlo simulations exhibit an O(N^-1/2) convergence rate. For the convergence study, we pick an M.
The program will run the simulations for N = 10^m, m = 1,...,M. After each simulation the error is computed. Finally, the error is plotted in a log-log plot against the number of simulations N.
The theoretical convergence rate of C*N^-1/2 is likewise plotted for a reference (this corresponds to a line with slope = -0.5 in the log-log plot).

We see that the simulations converge as expected with the antithetic variance reduction strategy being generally successful.

