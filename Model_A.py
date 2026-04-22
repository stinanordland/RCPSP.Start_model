
import pandas as pd
import pulp
from pathlib import Path
from datetime import datetime

# =========================
# 1. READ DATA
# =========================

# Folder path
BASE = r"C:\Users\stina\OneDrive\Documents\Master Science of Logistics\VÅR 2026\Data"

# =========================
# LOG FILE FOR THIS RUN
# =========================

log_dir = Path(BASE) / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = log_dir / f"run_log_{timestamp}.txt"

def log(*args, **kwargs):
    message = " ".join(str(arg) for arg in args)

    # print to console
    print(*args, **kwargs)

    # append same message to this run's log file
    with open(log_file, "a", encoding="utf-8") as f:
        print(message, file=f)

# Load Excel files
operations_df = pd.read_excel(f"{BASE}\\Operations.xlsx", sheet_name="Operations")
precedence_df = pd.read_excel(f"{BASE}\\PrecedenceOverview.xlsx", sheet_name="Job with process no ID")
capacity_df = pd.read_excel(f"{BASE}\\BiweeklyResourceData.xlsx", sheet_name="Original numbers")

# Remove rows with missing critical values
operations_df = operations_df.dropna(subset=["OP_NUM", "PROCESS_ID", "TOTAL_PROCESS_TIME"]).copy()
precedence_df = precedence_df.dropna(subset=["SEQ_NUM", "PRED_SEQ"]).copy()
capacity_df = capacity_df.dropna(subset=["Period"]).copy()

# Turn key columns into integer values
operations_df[["ITEM_NUMBER", "OP_NUM", "PROCESS_ID"]] = operations_df[
    ["ITEM_NUMBER", "OP_NUM", "PROCESS_ID"]
].apply(pd.to_numeric, errors="coerce").astype(int)

# Convert processing time to numeric values
operations_df["TOTAL_PROCESS_TIME"] = pd.to_numeric(
    operations_df["TOTAL_PROCESS_TIME"], errors="coerce"
)

# Convert precedence columns to integer values
precedence_df[["SEQ_NUM", "PRED_SEQ"]] = precedence_df[
    ["SEQ_NUM", "PRED_SEQ"]
].apply(pd.to_numeric, errors="coerce").astype(int)

# Convert capacity period to integer values
capacity_df["Period"] = pd.to_numeric(capacity_df["Period"], errors="coerce").astype(int)

# =========================
# 2. SETS
# =========================

# Number of time buckets available (T)
H = 50

# Set of time buckets  (T)
T = range(1, H + 1)

# Set of projects (I)
I = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

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
# Matches same-bucket precedence logic used in the model (start_m <= start_j).
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
model = pulp.LpProblem("Multi_Project_Bucket_Scheduling", pulp.LpMinimize)

# Index sets for binary decision variable (x^t_ij)
x_index = [(i, j, t) for i in I for j in J[i] for t in feasible_t[(i, j)]]

# Binary start-time variable (x^t_ij)
x = pulp.LpVariable.dicts("x", x_index, cat="Binary")

# Project completion time of project i
F = pulp.LpVariable.dicts("F", I, lowBound=0, upBound=H)

# Tardiness time of project i
L = pulp.LpVariable.dicts("L", I, lowBound=0, upBound=H)

# =========================
# 6. CONSTRAINTS
# =========================

# Each job must start exactly once
for i in I:
    for j in J[i]:
        model += pulp.lpSum(x[(i, j, t)] for t in feasible_t[(i, j)]) == 1

# Predecessor can not start 
for i in I:
    for j in J[i]:
        start_j = pulp.lpSum(t * x[(i, j, t)] for t in feasible_t[(i, j)])
        for m in P[j]:
            start_m = pulp.lpSum(t * x[(i, m, t)] for t in feasible_t[(i, m)])
            model += start_m <= start_j

# Resource consumption cannot exceed resource availability
for k in K:
    for t in T:
        model += pulp.lpSum(
            cons * x[(i, j, t)]
            for i in I
            for j, cons in jobs_by_resource[k]
            if t in feasible_t[(i, j)]
        ) <= a[(k, t)]

# Project completion time must be greater than or equal to finish time of every job
for i in I:
    for j in J[i]:
        model += F[i] >= pulp.lpSum(t * x[(i, j, t)] for t in feasible_t[(i, j)])

# Tardiness shall only capture late completion
for i in I:
    model += L[i] >= F[i] - D[i]

# Symmetry breaking
for i1, i2 in zip(list(I)[:-1], list(I)[1:]):
    if D[i1] == D[i2]:
        model += F[i1] <= F[i2]

# Fallback symmetry break for projects without same-due-date pairs
for i in range(1, len(I)):
    model += F[i] <= F[i+1]

# =========================
# 7. OBJECTIVE
# =========================

model += pulp.lpSum(L[i] for i in I)

# =========================
# 8. SOLVER
# =========================

# Defining solver settings
# msg=0 silences terminal solver output; logPath writes CBC output to file.
cbc_log_file = log_dir / f"Model_A_logfile_{timestamp}.log"
solver = pulp.PULP_CBC_CMD(
    msg=0,
    timeLimit=3200,
    options=["cuts on"],
    logPath=str(cbc_log_file),
)
solution_status = model.solve(solver)
log(f"CBC log written to: {cbc_log_file}")

# Convert status code to text and read objective value
status_text = pulp.LpStatus[solution_status]
objective_value = pulp.value(model.objective)

# Print summary results
log("Status:", status_text)
log("Total tardiness =", objective_value)

# =========================
# 9. OUTPUT
# =========================

output_file = f"{BASE}\\Model_A_{timestamp}.txt"

with open(output_file, "w", encoding="utf-8") as f:

    def out(*args, **kwargs):
        print(*args, file=f, **kwargs)

    def both(*args, **kwargs):
        print(*args, **kwargs)
        out(*args, **kwargs)

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
                f"Project {i:>2}: completion bucket = {pulp.value(F[i]):.2f}, "
                f"tardiness = {pulp.value(L[i]):.2f}, due bucket = {D[i]}"
            )

        both("\n============================================================")
        both("JOB ASSIGNMENTS BY PROJECT")
        both("============================================================")

        assigned = {}
        for i in I:
            assigned[i] = {}
            for j in J[i]:
                assigned[i][j] = next(
                    t for t in feasible_t[(i, j)]
                    if pulp.value(x[(i, j, t)]) is not None and pulp.value(x[(i, j, t)]) > 0.5
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
                    cons * pulp.value(x[(i, j, t)])
                    for i in I
                    for j, cons in jobs_by_resource[k]
                    if (i, j, t) in x and pulp.value(x[(i, j, t)]) is not None
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




