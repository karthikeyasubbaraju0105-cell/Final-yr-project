import os, math, random
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Optional Transformer predictor
# -------------------------------------------------------
try:
    from predictor import TransformerPredictor
    TRANSFORMER_AVAILABLE = True
except:
    TRANSFORMER_AVAILABLE = False

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
CFG = {
    "seed": 42,
    "num_users": 80,
    "num_edges": 20,
    "num_tasks": 5000,
    "time_horizon": 200,
    "bandwidth_mhz": 1.0,

    # edge CPU capacity
    "edge_resource_min": 20.0,
    "edge_resource_max": 40.0,

    # task distribution
    "task_size_min_mb": 5.0,
    "task_size_max_mb": 20.0,
    "task_cpu_min": 50.0,
    "task_cpu_max": 120.0,

    "user_speed_max": 15.0,

    # TORSE hyperparameters
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 0.1,
    "unit_cost": 0.1,
    "convergence_eps": 1e-3,
    "max_price_iters": 200,
    "price_init": 1.0,

    # cloud
    "cloud_resource": 500000.0,

    "coverage_radius": 500.0,

    "results_dir": "results",
}

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
os.makedirs(CFG["results_dir"], exist_ok=True)

# -------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------
@dataclass
class Task:
    user_id: int
    arrival_t: int
    size_mb: float
    cpu_need: float
    deadline: int = None

@dataclass
class User:
    uid: int
    x: float
    y: float
    speed: float
    tasks: List[Task] = field(default_factory=list)

@dataclass
class EdgeServer:
    eid: int
    x: float
    y: float
    resource_total: float
    resource_free: float
    price: float = CFG["price_init"]
    history: List[Tuple[int, float, float]] = field(default_factory=list)

@dataclass
class CloudServer:
    resource_total: float
    resource_free: float


# -------------------------------------------------------
# MATH HELPERS
# -------------------------------------------------------
def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def shannon_rate_mbps(bw_mhz, snr):
    return bw_mhz * 1e6 * math.log2(1 + snr) / 1e6

def delay_seconds(size_mb, rate_mbps):
    if rate_mbps <= 0:
        return float("inf")
    return (size_mb * 8.0) / rate_mbps


# -------------------------------------------------------
# GENERATORS
# -------------------------------------------------------
def generate_entities(cfg):
    users, edges = [], []

    for i in range(cfg["num_edges"]):
        x = np.random.uniform(0, 1000)
        y = np.random.uniform(0, 1000)
        cap = np.random.uniform(cfg["edge_resource_min"], cfg["edge_resource_max"])
        edges.append(EdgeServer(i, x, y, cap, cap, cfg["price_init"]))

    for u in range(cfg["num_users"]):
        x = np.random.uniform(0, 1000)
        y = np.random.uniform(0, 1000)
        sp = np.random.uniform(0, cfg["user_speed_max"])
        users.append(User(u, x, y, sp))

    cloud = CloudServer(cfg["cloud_resource"], cfg["cloud_resource"])
    return users, edges, cloud


def generate_tasks_for_users(users, cfg):
    """
    Generate EXACTLY cfg["num_tasks"] tasks.
    Arrival times are random within time_horizon.
    Users are chosen uniformly.
    """
    tasks = []

    for _ in range(cfg["num_tasks"]):
        u = random.choice(users)

        t = random.randint(0, cfg["time_horizon"] - 1)
        size = np.random.uniform(cfg["task_size_min_mb"], cfg["task_size_max_mb"])
        cpu = np.random.uniform(cfg["task_cpu_min"], cfg["task_cpu_max"])
        deadline = t + random.randint(5, 40)

        task = Task(u.uid, t, size, cpu, deadline)
        u.tasks.append(task)
        tasks.append(task)

    return tasks


# -------------------------------------------------------
# Predictor wrapper
# -------------------------------------------------------
class SimplePredictor:
    def __init__(self, k=3):
        self.k = k
    def predict_next_cpu(self, edge):
        hist = [h[1] for h in edge.history]
        return float(np.mean(hist[-self.k:])) if hist else 0.0

def build_predictor():
    if TRANSFORMER_AVAILABLE:
        try:
            return TransformerPredictor("edge_predictor.pt", seq_len=20)
        except:
            return SimplePredictor()
    return SimplePredictor()


# -------------------------------------------------------
# TORSE EXACT COMPETITIVENESS SELECTION
# -------------------------------------------------------
def tors_select(task, user, candidate_edges, cfg):
    prices = np.array([e.price for e in candidate_edges])
    ds = np.array([distance((user.x, user.y), (e.x, e.y)) for e in candidate_edges])

    ds_norm = ds / (ds.max() + 1e-9)

    alpha, beta, gamma = cfg["alpha"], cfg["beta"], cfg["gamma"]

    A = 1.0 / (alpha * prices + beta * ds_norm + 1e-9)
    prev = prices.copy()

    for _ in range(cfg["max_price_iters"]):
        prices = prices + gamma * (A - 1.0)
        A = 1.0 / (alpha * prices + beta * ds_norm + 1e-9)
        if np.max(np.abs(prices - prev)) < cfg["convergence_eps"]:
            break
        prev = prices.copy()

    for i, e in enumerate(candidate_edges):
        e.price = float(prices[i])

    idx = int(np.argmax(A))
    return candidate_edges[idx]


# -------------------------------------------------------
# MAIN SIMULATION
# -------------------------------------------------------
def run_simulation(cfg):
    users, edges, cloud = generate_entities(cfg)
    tasks = generate_tasks_for_users(users, cfg)
    predictor = build_predictor()

    metrics = {
        "total_tasks": len(tasks),
        "edge_success": 0,
        "cloud_success": 0,
        "failed_tasks": 0,
        "avg_delay_s_list": []
    }

    sim_logs = []

    for task in sorted(tasks, key=lambda t: t.arrival_t):

        user = users[task.user_id]
        pos = (user.x, user.y)

        # candidate edges
        candidates = [e for e in edges 
                      if distance(pos, (e.x, e.y)) <= cfg["coverage_radius"]]

        final_status = "failed"
        final_edge = None
        final_delay = None

        # Case 1 — no edges → try cloud only
        if not candidates:
            if cloud.resource_free >= task.cpu_need:
                cloud.resource_free -= task.cpu_need
                metrics["cloud_success"] += 1
                final_status = "cloud"
            else:
                metrics["failed_tasks"] += 1
            sim_logs.append({"t": task.arrival_t, "user": task.user_id,
                             "final_status": final_status,
                             "edge_id": final_edge,
                             "delay": final_delay})
            continue

        # TORSE pick
        edge = tors_select(task, user, candidates, cfg)
        needed = task.cpu_need

        # Case 2 — edge has enough resource
        if edge.resource_free >= needed:
            edge.resource_free -= needed
            edge.history.append((task.arrival_t, needed, task.size_mb))
            metrics["edge_success"] += 1
            final_status = "edge"

            d = distance(pos, (edge.x, edge.y))
            snr = 10 / (1 + (d / 100)**2)
            r = shannon_rate_mbps(cfg["bandwidth_mhz"], snr)
            final_delay = delay_seconds(task.size_mb, r)

        else:
            # Case 3 — try leasing deficit from cloud
            deficit = needed - edge.resource_free

            if cloud.resource_free >= deficit:
                cloud.resource_free -= deficit
                edge.resource_free = max(0,deficit) 

                edge.resource_free -= needed
                edge.history.append((task.arrival_t, needed, task.size_mb))

                metrics["edge_success"] += 1
                final_status = "edge"

                d = distance(pos, (edge.x, edge.y))
                snr = 10 / (1 + (d / 100)**2)
                r = shannon_rate_mbps(cfg["bandwidth_mhz"], snr)
                final_delay = delay_seconds(task.size_mb, r)

            else:
                metrics["failed_tasks"] += 1
                final_status = "failed"

        if final_delay is not None:
            metrics["avg_delay_s_list"].append(final_delay)

        sim_logs.append({
            "t": task.arrival_t,
            "user": task.user_id,
            "final_status": final_status,
            "edge_id": edge.eid if final_status=="edge" else None,
            "delay": final_delay
        })

    # summarizing
    avg_delay = float(np.mean(metrics["avg_delay_s_list"])) if metrics["avg_delay_s_list"] else float("nan")

    result = {
        "total_tasks": metrics["total_tasks"],
        "edge_success": metrics["edge_success"],
        "cloud_success": metrics["cloud_success"],
        "failed_tasks": metrics["failed_tasks"],
        "avg_delay_s": avg_delay,
        "cloud_remaining": cloud.resource_free,
    }

    # save logs
    df_logs = pd.DataFrame(sim_logs)
    df_logs.to_csv(os.path.join(cfg["results_dir"], "sim_logs.csv"), index=False)

    # save edge history (for Transformer training)
    hist_rows = []
    for e in edges:
        for (t, cpu, size) in e.history:
            hist_rows.append({"edge_id": e.eid, "t": t, "cpu": cpu})
    pd.DataFrame(hist_rows).to_csv(os.path.join(cfg["results_dir"], "edge_history.csv"), index=False)

    return result, df_logs, users, edges, cloud


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    print("Running TORS exact with cloud leasing (Milestone 4)...")

    res, logs, users, edges, cloud = run_simulation(CFG)

    print("RESULTS SUMMARY:")
    for k, v in res.items():
        print(f"  {k}: {v}")

    pd.DataFrame([res]).to_csv(os.path.join(CFG["results_dir"], "metrics.csv"), index=False)

    # plot remaining CPU on edges
    rem = [e.resource_free for e in edges]
    plt.figure(figsize=(8,4))
    plt.hist(rem, bins=12)
    plt.title("Edge remaining resources after run")
    plt.xlabel("Resource free")
    plt.ylabel("Count")
    plt.savefig(os.path.join(CFG["results_dir"], "edge_remaining_hist.png"))

    print("Saved outputs. Done.")
