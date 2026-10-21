import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import scipy as sp
import pyomo.environ as pyo
import os


from typing import Tuple, List, Dict

def cost(df: pd.DataFrame, power: float) -> float:
    a = df["a"].to_numpy()
    b = df["b"].to_numpy()
    return a * power + b * power ** 2

def scipy_min(df: pd.DataFrame, bounds: List[Tuple[float, float]], demand: float):

    # minimize calls fun(x, *args): x must be the first argument
    def total_cost(x: np.ndarray, df: pd.DataFrame) -> float:
        return cost(df, x).sum()

    def power_balance(x: np.array, demand: int) -> float:
        return x.sum() - demand

    constraint = {
        "type": "eq",
        "fun": power_balance,
        "args": (demand, )
    }

    result = sp.optimize.minimize(
        total_cost,
        x0 = np.zeros(len(df)),
        bounds = bounds,
        constraints = [constraint],
        # minimize calls the total_cost function with df as an additional argument (*args), so we need to pass it as a tuple
        args=(df,)
    )

    # print(f"Optimal power output: {result.x}")
    # print(f"Minimum cost: {result.fun}")

    return result.x, result.fun

def marginal_cost(df: pd.DataFrame, power: float) -> float:
    # remember: capped units do not set the price
    a = df["a"].to_numpy()
    b = df["b"].to_numpy()
    return a + 2 * b * power

def step_xy(df: pd.DataFrame)-> Tuple[List[float], List[float]]:

    x = [0] + list(df["cumulative_quantity"])
    y = list(df["price"]) + [df["price"].iloc[-1]] 

    return x, y

def plot_supply_demand(supply_df: pd.DataFrame, demand_df: pd.DataFrame):

    periods = sorted(supply_df["period"].unique())

    fig, ax = plt.subplots(1, len(periods), sharey=True, figsize=(12, 4), squeeze=False)
    ax = ax[0]

    for i, period in enumerate(periods):
        x_s, y_s = step_xy(supply_df[supply_df["period"] == period])
        x_d, y_d = step_xy(demand_df[demand_df["period"] == period])

        ax[i].step(x_s, y_s, where="post", label="Supply")
        ax[i].step(x_d, y_d, where="post", label="Demand")
        ax[i].set_title(f"Period {period}")
        ax[i].set_xlabel("Quantity (MW)")
        ax[i].set_ylabel("Price (€/MWh)")
        ax[i].legend()
        ax[i].grid()

    plt.tight_layout()
    plt.savefig("outputs/supply_demand.png")
    mplcursors.cursor(fig, hover=True) # show (x, y) of the curve under the mouse

def plot_supply_inelastic_demand(supply_df: pd.DataFrame, demand_df: pd.DataFrame):

    periods = sorted(supply_df["period"].unique())

    fig, ax = plt.subplots(1, len(periods), sharey=True, figsize=(12, 4), squeeze=False)
    ax = ax[0]

    for i, period in enumerate(periods):
        x_s, y_s = step_xy(supply_df[supply_df["period"] == period])
        # Inelastic demand: a vertical line at the total load
        total_demand = demand_df[demand_df["period"] == period]["quantity"].sum()

        ax[i].step(x_s, y_s, where="post", label="Supply")
        ax[i].axvline(total_demand, color="tab:orange", label=f"Demand ({total_demand:g} MW)")
        ax[i].set_title(f"Period {period}")
        ax[i].set_xlabel("Quantity (MW)")
        ax[i].set_ylabel("Price (€/MWh)")
        ax[i].legend()
        ax[i].grid()

    plt.tight_layout()
    plt.savefig("outputs/supply_inelastic_demand.png")
    mplcursors.cursor(fig, hover=True) # show (x, y) of the curve under the mouse

def optimization(period: int, supply_df: pd.DataFrame, demand_df: pd.DataFrame, duration: int) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()


    s = supply_df[supply_df["period"] == period]
    b = demand_df[demand_df["period"] == period]
    
    # Sets
    m.S = pyo.Set(initialize=range(0, len(s)))
    m.B = pyo.Set(initialize=range(0, len(b)))

    # Decision Variables Bounds
    def bounds_supplier(m,i):
        return (0, float(s.iloc[i]["quantity"]))
    def bounds_buyer(m,i):
        return (0, float(b.iloc[i]["quantity"]))
    
    # Decision Variables
    m.qs = pyo.Var(m.S, bounds=bounds_supplier)
    m.qb = pyo.Var(m.B, bounds=bounds_buyer)

    # OF
    # Maximize welfare - difference between buyers and suppliers
    def welfare(m) -> float:
        return sum(float(b.iloc[j]["price"]) * m.qb[j] for j in m.B) - sum(float(s.iloc[i]["price"]) * m.qs[i] for i in m.S)

    m.W = pyo.Objective(rule=welfare, sense=pyo.maximize)

    # Constraint
    # Energy sold = Energy bought
    def balance(m) -> bool:
        return sum(m.qb[j] for j in m.B) == sum(m.qs[i] for i in m.S)
    
    m.balance = pyo.Constraint(rule=balance)

    #Write LP
    os.makedirs("outputs", exist_ok=True)
    m.write(f"outputs/model_{period}.lp", io_options={"symbolic_solver_labels": True})
    
    #Solve
    m.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT) # import cause i want to receive the duals
    solver = pyo.SolverFactory("cplex_direct")
    res = solver.solve(m)


    # Build a df with results
    results_supplier_df = pd.DataFrame({
        "period": [period] * len(s),
        "supplier": s.index.values,
        "node": s["node"].values,
        "quantity offered": s["quantity"].values,
        "quantity accepted": [m.qs[i].value for i in m.S],
        "price offered": s["price"].values,
        "market price": m.dual.get(m.balance),
        "payment": [m.qs[i].value * m.dual.get(m.balance) * duration for i in m.S]
    })

    results_buyer_df = pd.DataFrame({
        "period": [period] * len(b),
        "buyer": b.index.values,
        "node": b["node"].values,
        "quantity offered": b["quantity"].values,
        "quantity accepted": [m.qb[j].value for j in m.B],
        "price offered": b["price"].values,
        "market price": m.dual.get(m.balance),
        "payment": [m.qb[j].value * m.dual.get(m.balance) * duration for j in m.B]
    })

    # os.makedirs("outputs", exist_ok=True)
    # results_supplier_df.to_csv(f"outputs/results_supplier_period_{period}.csv", index=False)
    # results_buyer_df.to_csv(f"outputs/results_buyer_period_{period}.csv", index=False)

    return results_supplier_df, results_buyer_df, m

def optimization_inealistic(period: int, supply_df: pd.DataFrame, demand_df: pd.DataFrame, duration: int):

    m = pyo.ConcreteModel()

    #Sets
    s = supply_df[supply_df["period"] == period]
    m.S = pyo.Set(initialize=range(0, len(s)))

    #Bounds
    def bounds_supplier(m, i):
        return (0, float(s.iloc[i]["quantity"]))

    #Decision Variables
    m.qs = pyo.Var(m.S, bounds=bounds_supplier)

    #OF
    #We only want to reduce costs on the supplier side
    def objective(m):
        return sum(s.iloc[j]["price"] * m.qs[j] for j in m.S)

    m.W = pyo.Objective(rule=objective, sense=pyo.minimize)

    #Constraints
    b = demand_df[demand_df["period"] == period]
    total_demand = b["quantity"].sum()

    def balance(m):
        return sum(m.qs[j] for j in m.S) == total_demand

    m.balance = pyo.Constraint(rule=balance)

    #Write LP
    os.makedirs("outputs", exist_ok=True)
    m.write(f"outputs/model_inealistic_{period}.lp", io_options={"symbolic_solver_labels": True})
    
    #Solve
    m.dual = pyo.Suffix(direction = pyo.Suffix.IMPORT) # import cause i want to receive the duals
    solver = pyo.SolverFactory("cplex_direct")
    res = solver.solve(m)

    # Build a df with results
    results_supplier_df = pd.DataFrame({
        "period": [period] * len(s),
        "supplier": s.index.values,
        "node": s["node"].values,
        "quantity offered": s["quantity"].values,
        "quantity accepted": [m.qs[i].value for i in m.S],
        "price offered": s["price"].values,
        "market price": m.dual.get(m.balance),
        "payment": [m.qs[i].value * m.dual.get(m.balance) * duration for i in m.S]
    })

    return results_supplier_df, m
    
if __name__ == "__main__":

    df = pd.read_csv("inputs/generators.csv", index_col="name")
    demand = pd.read_csv("inputs/demand.csv")["demand"].tolist()
    bids_df = pd.read_csv("inputs/bids.csv")
    lines_df = pd.read_csv("inputs/lines.csv")
    load_distribution_df = pd.read_csv("inputs/load_distribution.csv", index_col="node")

    S_BASE = 100
    X_PER_KM = 0.003 # pu
    U_KV = 220
    COS_PHI = 0.75


    # Cost at full power
    df["cost_fullpower"] = cost(df, df["pmax"])

    # Set generator bounds
    bounds = [(0, gen["pmax"]) for _, gen in df.iterrows()]
    # print(df.head(6))
    # print(bounds)

    # Scipy
    scipy_df = df.copy()
    for d in demand:
        power, total_cost = scipy_min(df, bounds, d)
        scipy_df[f"P_{d}"] = power
        scipy_df[f"cost_{d}"] = cost(df, power)
        scipy_df[f"total_cost_{d}"] = total_cost
        scipy_df[f"marginal_cost_{d}"] = marginal_cost(df, power)

    print("====================================================")
    print(f"Scipy optimization results for demands: {demand}")
    print(scipy_df)
    os.makedirs("outputs", exist_ok=True)
    scipy_df.to_csv("outputs/scipy_results.csv", index=False)

    # Map periods to demand
    map = {i+1: d for i, d in enumerate(demand)}
    bids_df["demand"] = bids_df["period"].map(map)

    selling_bids = bids_df[bids_df["direction"] == "sell"]
    selling_ordered_bids = selling_bids.sort_values(by=["period", "price"], ascending=[True, True])
    selling_ordered_bids["cumulative_quantity"] = selling_ordered_bids.groupby("period")["quantity"].cumsum()

    buying_bids = bids_df[bids_df["direction"] == "buy"]
    buying_ordered_bids = buying_bids.sort_values(by=["period", "price"], ascending=[True, False])
    buying_ordered_bids["cumulative_quantity"] = buying_ordered_bids.groupby("period")["quantity"].cumsum()

    plot_supply_demand(selling_ordered_bids, buying_ordered_bids)
    plot_supply_inelastic_demand(selling_ordered_bids, buying_ordered_bids)

    periods = bids_df["period"].unique()
    period_duration = {1: 15, 2: 3, 3: 6}


    print("====================================================")
    print("Running optimization with elastic demand (maximizing welfare)")

    results_supplier_df = {}
    results_buyer_df = {}
    m = {}

    for p in periods:
        results_supplier_df[p], results_buyer_df[p], m[p] = optimization(p, selling_ordered_bids, buying_ordered_bids, period_duration[p])
        print(f"Results for period {p}:")
        print(results_supplier_df[p])
        print(results_buyer_df[p])
        print(f"Duals for period {p}: {m[p].dual.get(m[p].balance)}")
        print(f"Quantity sold: {results_supplier_df[p]['quantity accepted'].sum()}")
        print(f"Quantity bought: {results_buyer_df[p]['quantity accepted'].sum()}")
        print(f"Total payment by buyers: {results_buyer_df[p]['payment'].sum()}")
        print(f"Total payment to suppliers: {results_supplier_df[p]['payment'].sum()}")
        print(f"Total welfare: {pyo.value(m[p].W)} €/h")
        print("\n")

    results_supplier_df_all = pd.concat(results_supplier_df.values(), ignore_index=True)
    results_buyer_df_all = pd.concat(results_buyer_df.values(), ignore_index=True)
    results_supplier_df_all.to_csv("outputs/results_supplier_all.csv", index=False)
    results_buyer_df_all.to_csv("outputs/results_buyer_all.csv", index=False)



    print("====================================================")
    print("Running optimization with inelastic demand (only minimizing costs on the supplier side)")
    results_supplier_inealistic_demand_df = {}
    m_inealistic = {}
    results_load_inealistic_demand_df = {}
    for p in periods:
        results_supplier_inealistic_demand_df[p], m_inealistic[p] = optimization_inealistic(
            period=p,
            supply_df=selling_ordered_bids,
            demand_df=buying_ordered_bids,
            duration=period_duration[p]
        )

        # Load at each node pays its share of the demand times the SMP
        smp = m_inealistic[p].dual.get(m_inealistic[p].balance)
        results_load_inealistic_demand_df[p] = pd.DataFrame({
            "period": p,
            "load": load_distribution_df["share"] / 100 * map[p],
            "market price": smp,
            "payment": load_distribution_df["share"] / 100 * map[p] * smp * period_duration[p]
        }).reset_index()

        print(f"Results for period {p}:")
        print(results_supplier_inealistic_demand_df[p])
        print(results_load_inealistic_demand_df[p])
        print(f"Duals for period {p}: {smp}")
        print(f"Quantity sold: {results_supplier_inealistic_demand_df[p]['quantity accepted'].sum()}")
        print(f"Total payment to suppliers: {results_supplier_inealistic_demand_df[p]['payment'].sum()}")
        print(f"Total payment by buyers: {results_load_inealistic_demand_df[p]['payment'].sum()}")
        # objective here is the cost of the accepted selling bids, not welfare (inelastic load has no bid value)
        print(f"Total generation cost: {pyo.value(m_inealistic[p].W)} €/h")
        print("\n")

    results_supplier_inealistic_demand_df_all = pd.concat(results_supplier_inealistic_demand_df.values(), ignore_index=True)
    results_supplier_inealistic_demand_df_all.to_csv("outputs/results_supplier_inealistic_demand_all.csv", index=False)
    results_load_inealistic_demand_df_all = pd.concat(results_load_inealistic_demand_df.values(), ignore_index=True)
    results_load_inealistic_demand_df_all.to_csv("outputs/results_load_inealistic_demand_all.csv", index=False)

 

    print("====================================================")
    print("DC Model")
    lines_df["max_power"] = np.sqrt(3) * U_KV * lines_df["Imax"] * COS_PHI / 1000
    lines_df["x"] = X_PER_KM * lines_df["km"]

    max_nodes = max(lines_df["from"].max(), lines_df["to"].max())
    b_matrix = np.zeros((max_nodes, max_nodes))

    for _, line in lines_df.iterrows():
        # each line is a literal whole line of the df
        i = int(line["from"]) - 1 # iterrows returns floats
        j = int(line["to"]) - 1

        b = 1 / line["x"]
        b_matrix[i,i] += b
        b_matrix[j,j] += b
        b_matrix[i,j] -= b
        b_matrix[j,i] -= b

    #Remove first line (0) and first column (0), which is because we're considering node 1 to be the referenc
    b_reduced = b_matrix[1:, 1:]

    generation_per_node = {}
    load_per_node = {}
    injection_per_node = {}
    injection_per_node_pu = {}
    nodes_df = {}
    theta = {}
    flow_df = {}

    for p in periods:
        generation_per_node[p] = results_supplier_df[p].groupby(by="node")["quantity accepted"].sum()
        load_per_node[p] = results_buyer_df[p].groupby(by="node")["quantity accepted"].sum()
        injection_per_node[p] = generation_per_node[p].sub(load_per_node[p], fill_value=0)
       
        
        # Convert from MW to pu
        injection_per_node_pu[p] = injection_per_node[p] / S_BASE
        
        theta[p] = np.linalg.solve(b_reduced, injection_per_node_pu[p].to_numpy()[1:]) # Removing reference from injection vector
        theta[p] = np.insert(theta[p], 0, 0)

        theta_from = theta[p][lines_df["from"]-1]
        theta_to = theta[p][lines_df["to"]-1]
        flow_df[p] = lines_df[["from", "to", "max_power"]].copy()
        flow_df[p].insert(0, "period", p)
        flow_df[p]["flow"] = (theta_from - theta_to) / lines_df["x"] * S_BASE
        flow_df[p]["within_limit"] = abs(flow_df[p]["flow"]) <= flow_df[p]["max_power"]
        flow_df[p]["overload%"] = abs(flow_df[p]["flow"]) / flow_df[p]["max_power"] * 100

        # pandas aligns the three Series by node; nodes missing in one get 0
    
        nodes_df[p] = pd.DataFrame({
            "period": p,
            "generation": generation_per_node[p],
            "load": load_per_node[p],
            "injection": injection_per_node[p],
            "theta": theta[p]
        }).fillna(0)

    nodes_all_df = pd.concat(nodes_df.values())
    flow_all_df = pd.concat(flow_df.values(), ignore_index=True)
    
    print("Computed each node's injected power and theta: ")
    print(nodes_all_df.head(20))
    print("\nFlow at each line")
    print(flow_all_df)
    # Open the interactive chart windows (hover only works here, not in the saved PNGs)
    plt.show()