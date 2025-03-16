# msbd6000m_a1

## Assignment Question

Consider the discrete-time asset allocation example in section 8.4 of Rao and Jelvis.  Suppose the single-time-step return of the risky asset as Y_t=a, prob = p, and b, prob = (1 - p). Suppose that T=10, use the TD method to find the Q function, and hence the optimal strategy.

## Key Assumptions

1. Instead of what is specified in section 8.4 of Rao and Jelvis, which states that the reward distribution after each time step follows a normal distribution, in this assignment the reward follows a bernoulli distribution, where there can only be 2 possible rewards, `a` or `b`.

## Formulation

- w_0: initial wealth
- t: discrete time steps
- w_t: wealth accumulated over t
- x_t: amount invested in risky asset at t
- w_t - x_t: amount invested in risk-free asset at t
- r: time-independent risk-free rate
- gamma: return discount rate

## Objective

- Maximize expectation on CARA Utility Function, which can be done by maximizing expected wealth at terminal state T

## How to run

1. Python runtime is 3.12.7.
2. Install dependencies with `pip install -r /path/to/requirements.txt`.
3. Read `main.ipynb` for different scenarios of risky asset stochastic return.

## How to run tests

The following command will run and list all tests:

```python
pytest -v
```

While the following will provide coverage report:

```python
pytest --cov=. --cov-report term-missing tests/
```
