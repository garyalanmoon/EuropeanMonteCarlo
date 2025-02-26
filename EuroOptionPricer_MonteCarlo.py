######## This code will price a European option via Monte Carlo simulation. The price is checked against the Black-Scholes formula and the error is reported
######## The stock price is assumed to follow a geometric Brownian motion with constant drift (r) and volatility (sig).
######## The stock price at expiry is generated directly (as opposed to simulating the entire price path) using the known solution to the GBM SDE:
######## S_t = S_0*exp( (r - 1/2*sig^2)t + sig*B_t ), B_t ~ N(0,t) a Brownian motion
######## We use B_t ~ sqrt(t)Z_t with Z_t ~ N(0,1)
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

####Parameters (Option, Market and Monte Carlo)
S_0 = 100 #current stock price
r = 0.05 #risk-free rate
sig = 0.2 #market volatility
K = 100 #strike price
T = 1 #time to maturity
q = 1 #option type parameter (use q = 1 for call and q = -1 for put)
N = 10**8 #number of sample paths to generate

def monte_carlo_price(S_0, K, r, vol, T, option_type, num_simulations):
    if option_type == 'call':
        q = 1
    else:
        q = -1

    np.random.seed(42)
    Z = np.random.randn(num_simulations)

    S_T = S_0 * np.exp( (r - 0.5 * vol**2)*T + vol * np.sqrt(T) * Z )

    payoffs = np.maximum(q * (S_T - K), 0)
    disc_payoffs = np.exp(-r * T) * payoffs
    V = np.mean(disc_payoffs)

    var = np.var(disc_payoffs, ddof=1) # Compute variance of payoffs
    std_error = np.sqrt(var/N) # Compute standard error of payoffs

    z_val = stats.norm.ppf(0.975) # Compute z score for 95% confidence assuming normal distribution
    ci_lower = V - z_val * std_error
    ci_upper = V + z_val * std_error

    return V, var, ci_lower, ci_upper

V, var, ci_lower, ci_upper = monte_carlo_price(S_0, K, r, sig, T, 'call', N)
print(f'The option price computed via Monte Carlo simulation is {V:.5f}')
print(f'The Monte Carlo simulation variance is {var:.5f}')
print(f'With 95% confidence the price lies between {ci_lower:.5f} and {ci_upper:.5f}')


#### Compute price via Black-Scholes formula
def black_scholes_price(S_0, K, r, vol, T, option_type):
    if option_type == 'call':
        q = 1
    else:
        q = -1

    d_1 = ( np.log(S_0/K) + ( r + 0.5 * sig**2 ) * T )/( sig * np.sqrt(T) )
    d_2 = d_1 - sig*np.sqrt(T)
    V_BS = q * ( S_0 * stats.norm.cdf(q*d_1) - K * np.exp(-r * T) * stats.norm.cdf(q*d_2) )

    return V_BS


####Check Error
def Euro_option_error(option_value, S_0, K, r, vol, T, option_type):
    if option_type == 'call':
        q = 1
    else:
        q = -1
    V_BS = black_scholes_price(S_0, K, r, sig, T, 'call')

    error = np.abs( option_value - V_BS )

    return error

error = Euro_option_error(V, S_0, K, r, sig, T, 'call')

print(f'The error via Monte Carlo simulation is {error:.5f}')


#### Apply antithetic variance reduction

def monte_carlo_price_anti_var_reduce(S_0, K, r, vol, T, option_type, num_simulations):
    if option_type == 'call':
        q = 1
    else:
        q = -1

    np.random.seed(42)
    Z = np.random.randn(int(num_simulations / 2) + 1)
    Z_anti = -Z

    S_T = S_0 * np.exp( (r - 0.5 * vol**2)*T + vol * np.sqrt(T) * Z )
    S_T_anti = S_0 * np.exp( (r - 0.5 * vol**2)*T + vol * np.sqrt(T) * Z_anti )

    payoffs = np.maximum(q * (S_T - K), 0)
    payoffs_anti = np.maximum(q * (S_T_anti - K), 0)

    effective_payoffs = 0.5 * payoffs + 0.5 * payoffs_anti

    disc_effective_payoffs = np.exp(-r * T) * effective_payoffs

    V_anti = np.mean(disc_effective_payoffs)

    var_anti = np.var(disc_effective_payoffs, ddof=1) # Compute variance of payoffs
    std_error_anti = np.sqrt(var_anti/N) # Compute standard error of payoffs

    z_val = stats.norm.ppf(0.975) # Compute z score for 95% confidence assuming normal distribution
    ci_lower_anti = V_anti - z_val * std_error_anti
    ci_upper_anti = V_anti + z_val * std_error_anti

    return V_anti, var_anti, ci_lower_anti, ci_upper_anti

V_anti, var_anti, ci_lower_anti, ci_upper_anti = monte_carlo_price_anti_var_reduce(S_0, K, r, sig, T, 'call', N)

print(f'The option price computed via Monte Carlo simulation with antithetic variance reduction is {V_anti:.5f}')

print(f'The variance in Monte Carlo simulation with antithetic variance reduction is {var_anti:.5f}')

print(f'With 95% confidence the price lies between {ci_lower_anti:.5f} and {ci_upper_anti:.5f}')

error_anti = Euro_option_error(V_anti, S_0, K, r, sig, T, 'call')

print(f'The error via Monte Carlo simulation with antithetic variance reduction is {error_anti:.5f}')


#### Run convergence study

def monte_carlo_convergence_study(M):
    errors = np.zeros(M)
    errors_anti = np.zeros(M)
    N_vals = np.zeros(M)
    for m in range(M):
        N_vals[m] = 10 ** (m+1)
    for m in range(M):
        V, var, ci_lower, ci_upper = monte_carlo_price(S_0, K, r, sig, T, 'call', 10** (m+1))
        errors[m] = Euro_option_error(V, S_0, K, r, sig, T, 'call')

        V_anti, var_anti, ci_lower_anti, ci_upper_anti = monte_carlo_price_anti_var_reduce(S_0, K, r, sig, T, 'call', 10 ** (m+1))
        errors_anti[m] = Euro_option_error(V_anti, S_0, K, r, sig, T, 'call')

    C=10 # Scaling factor for theoretical convergence line
    plt.figure(figsize=(10,6))
    plt.loglog(N_vals, errors, 'o-', label='Monte Carlo Error')
    plt.loglog(N_vals, errors_anti, 's-', label='Monte Carlo with Antithetic Variance Reduction Error')
    plt.loglog(N_vals, C * (1/(N_vals**0.5)), '--', label='Theoretical O(N^-1/2) Convergence')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Error in Monte Carlo Price')
    plt.title('Monte Carlo Pricing with Variance Reduction Convergence Study (log-log plot)')
    plt.legend()

    return plt.show()

monte_carlo_convergence_study(8)
