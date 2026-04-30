import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from datetime import datetime

# =========================
# 1. READ DATA
# =========================

# Folder path
BASE = r"C:\Users\stina\OneDrive\Documents\Master Science of Logistics\SPRING 2026\Data"

# =========================
# LOG FILE FOR THIS RUN
# =========================

log_dir = Path(BASE) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_dir / f"run_log_{timestamp}.txt"
solver_log_file = log_dir / f"Model_A__{timestamp}.log"

def log(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)

    # append same message to this run's log file only (no console output)
    with open(log_file, "a", encoding="utf-8") as f:
        print(message, file=f)

# Load Excel files
operations_df = pd.read_excel(f"{BASE}\\Operations.xlsx", sheet_name="Operations")
precedence_df = pd.read_excel(f"{BASE}\\PrecedenceOverview.xlsx", sheet_name="Job with process no ID")
capacity_df = pd.read_excel(f"{BASE}\\BiweeklyResourceData.xlsx", sheet_name="Original numbers")

# Remove rows with missing critical values
operations_df = operations_df.dropna(subset=["OP_NUM", "PROCESS_ID", "TOTAL_PROCESS_TIME"]).copy()
precedence_df = precedence_df.dropna(subset=["SEQ_NUM", "PRED_SEQ"]).copy()

# Convert only numeric columns
operations_df[["ITEM_NUMBER", "OP_NUM"]] = operations_df[
    ["ITEM_NUMBER", "OP_NUM"]
].apply(pd.to_numeric, errors="coerce").astype(int)

# Keep PROCESS_ID as string
operations_df["PROCESS_ID"] = operations_df["PROCESS_ID"].astype(str).str.strip()

# Convert processing time to numeric values
operations_df["TOTAL_PROCESS_TIME"] = pd.to_numeric(
    operations_df["TOTAL_PROCESS_TIME"], errors="coerce"
)

# Convert precedence columns to integer values
precedence_df[["SEQ_NUM", "PRED_SEQ"]] = precedence_df[
    ["SEQ_NUM", "PRED_SEQ"]
].apply(pd.to_numeric, errors="coerce").astype(int)

# Convert Period to numeric first
capacity_df["Period"] = capacity_df["Period"].astype(str).str.strip()

# Remove rows where Period is missing or blank
capacity_df = capacity_df[
    capacity_df["Period"].notna() & (capacity_df["Period"] != "")
].copy()

# =========================
# 2. SETS
# =========================

# Number of time buckets available (T)
H = 50

# Set of time buckets  (T)
T = range(1, H + 1)

# Set of projects (I)
I = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

# Number of jobs (J)
J_common = sorted(operations_df["OP_NUM"].unique())

# Set of jobs (J)
J = {i: J_common for i in I}

# Set of resources (k)
K = sorted(operations_df["PROCESS_ID"].unique())

# Set of predecessors for job j (P_j)
P = precedence_df.groupby("SEQ_NUM")["PRED_SEQ"].apply(
    lambda s: sorted(set(int(v) for v in s if pd.notna(v)))
).to_dict()
P = {j: [m for m in P.get(j, []) if m != j] for j in J_common}

# =========================
# 3. PARAMETERS
# =========================

# Due dates for projects (D_i)
D = {
    1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11,
    7: 18, 8: 18, 9: 18, 10: 18, 11: 18, 12: 18, 13: 18, 14: 18,
    15: 24, 16: 24, 17: 24, 18: 24, 19: 24,
    20: 36, 21: 36, 22: 36
}

# Release time for each project (r_i)
r = {i: 1 for i in I}

# Processing time for each job (d_ij) (Found in Excel-file)
d = operations_df.groupby("OP_NUM")["TOTAL_PROCESS_TIME"].sum().astype(float).to_dict()

# Resource consumption of resource k to perform job j (c^k_j)
c = operations_df.groupby(["OP_NUM", "PROCESS_ID"])["TOTAL_PROCESS_TIME"].sum().astype(float).to_dict()

# Capacity data by period
capacity_lookup = capacity_df.set_index("Period")

# Resource availability of k in time bucket T (a_kt)
a = {
    (k, t): float(capacity_lookup.at[k, str(t)])
    if k in capacity_lookup.index and str(t) in capacity_lookup.columns
    else 0.0
    for k in K for t in T
}

# =========================
# 4. PREPROCESSING BLOCK
# =========================

# Jobs consuming each resource
jobs_by_resource = {
    k: [(j, cons) for (j, kk), cons in c.items() if kk == k and cons > 0]
    for k in K
}

# Earliest possible bucket from precedence constraints
ES = {j: 1 for j in J_common}
changed = True
while changed:
    changed = False
    for j in J_common:
        if P[j]:
            es_new = max(ES[m] for m in P[j])
            if es_new > ES[j]:
                ES[j] = es_new
                changed = True

log(f"Max ES: {max(ES.values())}")

# Feasible start buckets for each project-job pairing
feasible_t = {
    (i, j): list(range(max(r[i], ES[j]), H + 1))
    for i in I for j in J[i]
}

# =========================
# 5. MODEL
# =========================

# Creating the optimization model
env = gp.Env(empty=True)
env.setParam("LogToConsole", 0)
env.setParam("LogFile", str(solver_log_file))
env.start()
model = gp.Model("Multi_Project_Bucket_Scheduling", env=env)

# Index sets for binary decision variable (x^t_ij)
x_index = [(i, j, t) for i in I for j in J[i] for t in feasible_t[(i, j)]]

# Binary start-time variable (x^t_ij)
x = {}
for i, j, t in x_index:
    x[(i, j, t)] = model.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name=f"x_{i}_{j}_{t}")

# Project completion time of project i
F = {}
for i in I:
    F[i] = model.addVar(lb=0, ub=H, vtype=GRB.INTEGER, name=f"F_{i}")

# Tardiness time of project i
L = {}
for i in I:
    L[i] = model.addVar(lb=0, ub=H, vtype=GRB.INTEGER, name=f"L_{i}")

# =========================
# 6. CONSTRAINTS
# =========================

# Each job must start exactly once
for i in I:
    for j in J[i]:
        model.addConstr(
            gp.quicksum(x[(i, j, t)] for t in feasible_t[(i, j)]) == 1
        )

# Predecessor can not start 
for i in I:
    for j in J[i]:
        start_j = gp.quicksum(t * x[(i, j, t)] for t in feasible_t[(i, j)])
        for m in P[j]:
            start_m = gp.quicksum(t * x[(i, m, t)] for t in feasible_t[(i, m)])
            model.addConstr(start_m <= start_j)

# Resource consumption cannot exceed resource availability
for k in K:
    for t in T:
        model.addConstr(
            gp.quicksum(
                cons * x[(i, j, t)]
                for i in I
                for j, cons in jobs_by_resource[k]
                if t in feasible_t[(i, j)]
            ) <= a[(k, t)]
        )

# Project completion time must be greater than or equal to finish time of every job
for i in I:
    for j in J[i]:
        model.addConstr(F[i] >= gp.quicksum(t * x[(i, j, t)] for t in feasible_t[(i, j)]))

# Tardiness shall only capture late completion
for i in I:
    model.addConstr(L[i] >= F[i] - D[i])

# Symmetry breaking
for i1, i2 in zip(list(I)[:-1], list(I)[1:]):
    if D[i1] == D[i2]:
        model.addConstr(F[i1] <= F[i2])

# Fallback symmetry break for projects without same-due-date pairs
for i in range(1, len(I)):
    model.addConstr(F[i] <= F[i+1])

# =========================
# 7. OBJECTIVE
# =========================

model.setObjective(gp.quicksum(L[i] for i in I), GRB.MINIMIZE)

# =========================
# 8. SOLVER
# =========================

# Update the model before solving
model.update()

# Set Gurobi parameters
model.Params.TimeLimit = 3600
model.Params.MIPGap = 0.0
model.Params.OutputFlag = 1
model.Params.LogToConsole = 0

# Optimize the model
model.optimize()

# Convert status code to text and read objective value
status_code = model.status
if status_code == GRB.OPTIMAL:
    status_text = "Optimal"
elif status_code == GRB.SUBOPTIMAL:
    status_text = "Feasible"
else:
    status_text = "Infeasible/Unbounded"

objective_value = model.ObjVal if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL] else None

# Print summary results
log("Status:", status_text)
if objective_value is not None:
    log("Total tardiness =", objective_value)
else:
    log("Total tardiness = No solution found")

# =========================
# 9. OUTPUT
# =========================

output_file = f"{BASE}\\Clean_model_output_{timestamp}.txt"

with open(output_file, "w", encoding="utf-8") as f:

    def out(*args, **kwargs):
        print(*args, file=f, **kwargs)

    def both(*args, **kwargs):
        out(*args, **kwargs)

    def clean_zero(value, tol=1e-6):
        return 0.0 if abs(value) < tol else value

    both("============================================================")
    both("SOLUTION STATUS")
    both("============================================================")
    both(status_text)

    if status_text in ["Optimal", "Feasible"]:

        both("\n============================================================")
        both("OBJECTIVE VALUE")
        both("============================================================")
        both(f"Total tardiness = {objective_value:.2f}")

        both("\n============================================================")
        both("PROJECT RESULTS")
        both("============================================================")
        for i in I:
            both(
                f"Project {i:>2}: completion bucket = {clean_zero(F[i].X):.2f}, "
                f"tardiness = {clean_zero(L[i].X):.2f}, due bucket = {D[i]}"
            )

        both("\n============================================================")
        both("JOB ASSIGNMENTS BY PROJECT")
        both("============================================================")

        assigned = {}
        for i in I:
            assigned[i] = {}
            for j in J[i]:
                # Robust fallback: pick the bucket with the highest x-value.
                # This avoids issues if the solver returns a fractional solution.
                candidates = [
                    (t, x[(i, j, t)].X or 0.0)
                    for t in feasible_t[(i, j)]
                ]
                assigned_t, assigned_val = max(candidates, key=lambda item: item[1])
                assigned[i][j] = assigned_t

                if assigned_val < 0.5:
                    both(
                        f"Warning: Project {i}, Job {j} has no near-binary start (max x = {assigned_val:.4f}). "
                        f"Using bucket {assigned_t}."
                    )

        for i in I:
            both(f"\nProject {i}")
            for j, t in sorted(assigned[i].items(), key=lambda item: (item[1], item[0])):
                both(f"  Bucket {t:>2} <- Job {j:>4} (proc_time = {d[j]:.2f})")


        both("\n============================================================")
        both("RESOURCE USAGE BY BUCKET")
        both("============================================================")

        for k in K:
            printed_any = False
            for t in T:
                used = sum(
                    cons * x[(i, j, t)].X
                    for i in I
                    for j, cons in jobs_by_resource[k]
                    if (i, j, t) in x and x[(i, j, t)].X is not None
                )
                if used > 1e-6:
                    both(
                        f"Resource {k:>4}, bucket {t:>2}: "
                        f"used {used:>8.2f}, available {a[(k, t)]:>8.2f}"
                    )
                    printed_any = True
            if printed_any:
                both()

        both("\n============================================================")
        both("JOB START SUMMARY TABLE")
        both("============================================================")
        both("Project   Job   StartBucket   ProcTime   FinishMeasure")
        both("-------   ---   -----------   --------   -------------")

        for i in I:
            for j, t in sorted(assigned[i].items(), key=lambda item: (item[1], item[0])):
                both(
                    f"{i:>7}   {j:>3}   {t:>11}   "
                    f"{d[j]:>8.2f}   {t + d[j]:>13.2f}"
                )

    else:
        both("\nNo integer-feasible solution was found.")
        both("Therefore there are no valid job-bucket assignments to print.")
        both("Any objective value shown is only the LP relaxation value.")

log(f"\nResults written to: {output_file}")
log(f"Run log written to: {log_file}")
log(f"Gurobi solver log written to: {solver_log_file}")


