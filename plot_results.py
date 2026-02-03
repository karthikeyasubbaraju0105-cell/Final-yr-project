import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("results/alg_comparison.csv")
agg = df.groupby("algorithm").mean().reset_index()
plt.figure(figsize=(8,5))
plt.bar(agg["algorithm"], agg["edge_success"]/agg["total_tasks"], label="edge_success_rate")
plt.ylabel("Edge success rate")
plt.title("Algorithm comparison - average over runs")
plt.savefig("results/alg_compare_edge_success.png")
plt.show()
