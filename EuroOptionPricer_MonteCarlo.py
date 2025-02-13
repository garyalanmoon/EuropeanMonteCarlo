######## This code will price a European option via Monte Carlo simulation. The price is checked against the Black-Scholes formula and the error is reported
######## The stock price is assumed to follow a geometric Brownian motion with constant drift (r) and volatility (sig).
######## The stock price at expiry is generated directly (as opposed to simulating the entire price path) using the known solution to the GBM SDE:
######## S_t = S_0*exp( (r - 1/2*sig^2)t + sig*B_t ), B_t ~ N(0,t) a Brownian motion
######## We use B_t ~ sqrt(t)Z_t with Z_t ~ N(0,1)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

####Parameters (Option, Market and Monte Carlo)
S_0 = 100 #current stock price
r = 0.05 #risk-free rate
sig = 0.2 #market volatility
K = 100 #strike price
T = 1 #time to maturity
q = -1 #option type parameter (use q = 1 for call and q = -1 for put)
N = 10**4 #number of sample paths to generate

####Setup for Monte Carlo
np.random.seed(42)
Z = np.random.randn(N)

####Generate Terminal Stock Prices (assume GBM with constant vol and rates)
S_T = S_0 * np.exp( (r - 0.5 * sig**2)*T + sig * np.sqrt(T) * Z )

####Compute Option Payoff
payoffs = np.maximum(q*(S_T - K),0)

####Price the Option
discPayoffs = np.exp(-r*T)*payoffs
V = np.mean(discPayoffs)
print(V)

####Price the Option via BS Formula
d_1 = ( np.log(S_0/K) + ( r + 0.5 * sig**2 ) * T )/( sig * np.sqrt(T) )
d_2 = d_1 - sig*np.sqrt(T)
V_BS = q * ( S_0 * norm.cdf(q*d_1) - K * np.exp(-r * T) * norm.cdf(q*d_2) )

####Check Error
error = np.abs( V - V_BS )
print(error)
