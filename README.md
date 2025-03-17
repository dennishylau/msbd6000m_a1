# msbd6000m_a1

## Assignment Scope

Please see `Report.pdf` for details.

## Formulation

- `wealth`: wealth is initialized at 1.0
- `t`: discrete time steps with terminal state `T`
- `risky_asset`: a bernoulli distribution with probability `p` returning yield `a` else `b`
- `action`/`risky_alloc`: percentage of wealth with range [0,1] to be placed in `risky_asset` at particular time step
- `risk_free_asset`: percentage of wealth allocated will be at `1 - risky_alloc` in particular time step
- `r`: time-independent risk-free yield as returned by `risk_free_asset`
- `gamma`: MDP discount rate
- `reward_eval`: the CARA Utility Function

## Objective

- Maximize expectation on CARA Utility Function, which can be done by maximizing expected wealth at terminal state T.

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
