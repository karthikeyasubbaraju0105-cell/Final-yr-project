# run_experiments.py
# Run TORS exact and baselines, aggregate results
import os, shutil, copy
import pandas as pd
import matplotlib.pyplot as plt

from run_simulation import run_simulation, CFG, generate_entities, generate_tasks_for_users, shannon_rate_mbps
import baselines

def run_baseline_alg(alg, cfg, run_id=0):
    users, edges, cloud = generate_entities(cfg)
    tasks = generate_tasks_for_users(users, cfg)
    metrics = {"total_tasks": len(tasks), "edge_success":0, "cloud_success":0, "failed_tasks":0, "delays":[]}
    logs=[]
    for task in sorted(tasks, key=lambda t: t.arrival_t):
        if alg=="gca":
            status, eid, r = baselines.gca_allocate(task, users, edges, cfg)
        elif alg=="mvac":
            status, eid, r = baselines.mvac_allocate(task, users, edges, cfg, cloud)
        elif alg=="dpka":
            status, eid, r = baselines.dpka_allocate(task, users, edges, cfg)
        else:
            status,eid,r = ("none", None, None)

        if status=="edge":
            metrics["edge_success"] += 1
            edge = next(e for e in edges if e.eid==eid)
            d = ((users[task.user_id].x-edge.x)**2 + (users[task.user_id].y-edge.y)**2)**0.5
            snr = 10.0 / (1.0 + (d/100.0)**2)
            rate = shannon_rate_mbps(cfg["bandwidth_mhz"], snr)
            delay = transmission_delay_seconds(task.size_mb, rate)
            metrics["delays"].append(delay)
        elif status=="cloud":
            metrics["cloud_success"] += 1
        else:
            metrics["failed_tasks"] += 1
        logs.append({"t":task.arrival_t, "user":task.user_id, "final_status":status, "edge_id":eid, "delay": (metrics["delays"][-1] if metrics["delays"] else None)})
    res = {"total_tasks": metrics["total_tasks"], "edge_success":metrics["edge_success"], "cloud_success":metrics["cloud_success"], "failed_tasks":metrics["failed_tasks"], "avg_delay_s": (pd.Series(metrics["delays"]).mean() if metrics["delays"] else float('nan'))}
    return res, logs

def main():
    base_cfg = copy.deepcopy(CFG)
    base_cfg["num_users"] = 80
    base_cfg["num_edges"] = 20
    base_cfg["time_horizon"] = 200

    algorithms = ["tors_exact", "gca", "mvac", "dpka"]
    results = []
    runs_per_alg = 3

    for alg in algorithms:
        for r in range(runs_per_alg):
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = 100 + r
            cfg["results_dir"] = f"results/{alg}_run{r}"
            os.makedirs(cfg["results_dir"], exist_ok=True)
            if alg=="tors_exact":
                res, logs, users, edges, cloud = run_simulation(cfg)
            else:
                res, logs = run_baseline_alg(alg, cfg, r)
            res["algorithm"] = alg
            res["run"] = r
            results.append(res)
            print(f"Completed {alg} run {r}: {res}")

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/alg_comparison.csv", index=False)
    print("Saved results/alg_comparison.csv")

    # simple plot: edge success rate
    df2 = df.groupby("algorithm").mean().reset_index()
    df2["edge_success_rate"] = df2["edge_success"] / df2["total_tasks"]
    plt.figure(figsize=(8,5))
    plt.bar(df2["algorithm"], df2["edge_success_rate"])
    plt.title("Edge success rate (mean over runs)")
    plt.ylabel("Edge success rate")
    plt.savefig("results/alg_compare_edge_success.png")
    print("Saved results/alg_compare_edge_success.png")

if __name__ == "__main__":
    main()
