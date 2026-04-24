import numpy as np

class PSACG:
    def __init__(self, users, resources, k_initial=500.0, epsilon=250): #Constructor
        """
        users: {List of dicts {'id', 'D_i', 'R_ir', 'd_ir', 'P_i', 'weights'}
        resources: List of dicts {'id', 'f_r', 'p_r', 'T_proc'}
        """
        self.users = users
        self.resources = resources
        self.epsilon = epsilon
        self.k = k_initial
        self.c = 3e8  # Speed of light
        
        # State: x[user_idx][res_idx] = amount of data offloaded from user to that resource
        self.x = np.zeros((len(users), len(resources)))
        # Total load on each resource e
        self.xe = np.zeros(len(resources))
    
    def get_marginal_cost(self, i_idx, r_idx, is_add=True):
        if is_add:
            return self.get_totalcost(i_idx, r_idx, True) - self.get_totalcost(i_idx, r_idx,no_add= True) 
               #Cost(xie+k,xe+k) - Cost(xie,xe) for addition
        else:
            return self.get_totalcost(i_idx, r_idx, no_add=True) - self.get_totalcost(i_idx, r_idx, False) 
               #Cost(xie,xe) - Cost(xie-k,xe-k) for addition
    
    def get_totalcost(self, i_idx, r_idx, is_add=True,no_add=False):
        user = self.users[i_idx]
        res = self.resources[r_idx]
        
        # Pull USER-SPECIFIC weights
        alpha = user['weights']['alpha']
        beta = user['weights']['beta']
        gamma = user['weights']['gamma']
        
        test_xe = self.xe[r_idx]
        test_x = self.x[i_idx, r_idx]
        # Adjust load for the hypothetical move
        if not no_add:
          test_xe = self.xe[r_idx] + (self.k if is_add else -self.k)
          test_x=self.x[i_idx, r_idx] + (self.k if is_add else -self.k)
        
        # 1. Latency Components
        t_trans = test_x/ user['R_ir'][r_idx]
        
        t_prop = user['d_ir'][r_idx] / self.c
        
        # Congestion Term: Denominator cannot be <= 0
        remaining_cap = res['f_r'] - test_xe
        if remaining_cap <= 0:
            return float('inf') # Infinite cost if capacity exceeded
            
        t_queue = 1.0 / remaining_cap
        t_total = t_trans + t_prop + t_queue + res['T_proc']
        
        # 2. Energy Component (User Battery Drain)
        e_user = user['P_i'] * t_trans
        
        # 3. Price Component
        price = res['p_r'] * test_x
        
        return (alpha * t_total) + (beta * e_user) + (gamma * price)

    def add_packet(self, i_idx):
        costs = [self.get_marginal_cost(i_idx, r, True) for r in range(len(self.resources))]
        best_r = np.argmin(costs)
        for r in range(len(self.resources)):
            print(f"Marginal Cost of each resource on adding k: {self.get_marginal_cost(i_idx,r,True)}")
        self.x[i_idx, best_r] += self.k
        self.xe[best_r] += self.k
        print(f"Resource added to:{best_r}, Current load on resource: {self.xe[best_r]}")
        #self.restore_equilibrium()

    def restore_equilibrium(self):
        shuffled = True
        while shuffled:
            shuffled = False
            for i in range(len(self.users)):
                used_res = np.where(self.x[i] > 0)[0]
                
                if len(used_res) == 0: continue
                print(f"User{i}")
                # Best backward move (to leave)
                
                print(f"Maximum cost of each resource on leaving k: {[float(self.get_marginal_cost(i, r, False)) for r in used_res]}")
                b_costs = {r: self.get_marginal_cost(i, r, False) for r in used_res}
                r_max = max(b_costs, key=b_costs.get)
                
                print(f"Minimum cost of each resource on joining k: {[float(self.get_marginal_cost(i, r, True)) for r in range(len(self.resources))]}")
                # Best forward move (to join)
                f_costs = [self.get_marginal_cost(i, r, True) for r in range(len(self.resources))]
                r_min = np.argmin(f_costs)
                
                
                if f_costs[r_min] < b_costs[r_max]:
                    print(f"User {i} - Best leave: Resource {r_max} with cost {float(b_costs[r_max])}, Best join: Resource {r_min} with cost {float(f_costs[r_min])}")
                    self.x[i, r_max] -= self.k
                    self.xe[r_max] -= self.k
                    self.x[i, r_min] += self.k
                    self.xe[r_min] += self.k
                    shuffled = True
                else:
                    print("Not beneficial to move.")
                print(f"User {i} - Current load: {self.x[i]}")        
                    

    def run(self):
        print("Addition Phase:")
        for i in range(len(self.users)):
            remaining_data = self.users[i]['D_i']
            while remaining_data > 0:
                self.add_packet(i)
                remaining_data -= self.k
            print(f"User {i} - Total offloaded: {self.x[i]}")    
                
        print("Equilibrium Restoration Phase:")
        while self.k > self.epsilon:
            self.k /= 2.0
            self.restore_equilibrium()
            
        return self.x / np.sum(self.x, axis=1)[:, None]


if __name__ == "__main__":
    # Normalized weights for gamers and IoT devices
    gamer_weights = {'alpha': 0.8, 'beta': 0.1, 'gamma': 0.1}
    iot_weights = {'alpha': 0.1, 'beta': 0.7, 'gamma': 0.1}

    users = [
        {'id': 'Gamer1', 'D_i': 500, 'R_ir': [50, 10], 'd_ir': [1, 2000], 'P_i': 0.5, 'weights': gamer_weights},
        {'id': 'IoT1',   'D_i': 500, 'R_ir': [50, 10], 'd_ir': [1, 2000], 'P_i': 0.1, 'weights': iot_weights},
        {'id': 'Gamer2', 'D_i': 500, 'R_ir': [50, 10], 'd_ir': [1, 2000], 'P_i': 0.5, 'weights': gamer_weights},
        {'id': 'IoT2',   'D_i': 500, 'R_ir': [50, 10], 'd_ir': [1, 2000], 'P_i': 0.1, 'weights': iot_weights}
    ]
    
    resources = [
        {'id': 'UAV_1', 'f_r':900, 'p_r': 0.01, 'T_proc': 0.01},
        {'id': 'SAT_1', 'f_r': 2000, 'p_r': 0.01, 'T_proc': 0.05},
        
    ]
    solver = PSACG(users, resources)
    final_fractions = solver.run()
    
    print("Final Offloading Fractions (Nash Equilibrium):")
    for idx, user in enumerate(users):
        print(f"{user['id']}: {final_fractions[idx]}")