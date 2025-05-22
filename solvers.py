from scipy.optimize import linprog

def solve_break_even(data):
    fc = data["fixed_cost"]
    vc = data["variable_cost_per_unit"]
    price = data["selling_price"]
    bep = fc / (price - vc)
    return {
        "break_even_point": round(bep, 2),
        "explanation": f"Break-even = {fc} / ({price} - {vc}) = {round(bep, 2)}"
    }

def solve_transportation(data):
    c = [item for row in data["cost_matrix"] for item in row]
    m, n = len(data["supply"]), len(data["demand"])

    A_eq = []
    b_eq = []

    for i in range(m):
        row = [0] * (m * n)
        for j in range(n):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(data["supply"][i])

    for j in range(n):
        row = [0] * (m * n)
        for i in range(m):
            row[i * n + j] = 1
        A_eq.append(row)
        b_eq.append(data["demand"][j])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq)
    solution_matrix = res.x.reshape(m, n)
    return {
        "total_cost": res.fun,
        "solution_matrix": solution_matrix.tolist(),
        "explanation": f"Tổng chi phí tối ưu là {round(res.fun, 2)}"
    }
