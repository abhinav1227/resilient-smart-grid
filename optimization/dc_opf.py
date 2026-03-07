import cvxpy as cp
import numpy as np

def solve_load_shedding(load_p, gen_p_max, solver='OSQP'):
    n_buses = len(load_p)
    shed = cp.Variable(n_buses, nonneg=True)
    gen = cp.Variable(n_buses, nonneg=True)
    objective = cp.Minimize(cp.sum(shed))
    constraints = [
        cp.sum(gen) + cp.sum(shed) == cp.sum(load_p),
        gen <= gen_p_max,
        shed <= load_p,
        shed >= 0,
        gen >= 0
    ]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver)
    except:
        prob.solve()  # fallback to default solver
    if prob.status == 'optimal':
        return shed.value, gen.value
    else:
        return None, None