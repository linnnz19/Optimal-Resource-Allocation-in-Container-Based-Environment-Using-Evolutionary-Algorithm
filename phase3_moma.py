# To run: python phase3_moma.py -i device_spec.cfg

import yaml
import optparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import random

from MomaIndividual import MomaIndividual, Worker
from MomaPopulation import MomaPopulation

# ---------------------------------------------------------
# Config Loader
# ---------------------------------------------------------
class MomaConfig:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            yml = yaml.safe_load(f)
        self.dev_specs = yml.get('DeviceSpecs', {})
        self.sim = yml.get('Simulation', {})
        self.moma = yml.get('MOMA', {}) 
        
        self.seed = self.sim.get('randomSeed', 42)
        self.pop_size = self.moma.get('populationSize', 50)
        self.n_gen = self.moma.get('generationCount', 40)
        self.cx_prob = self.moma.get('crossoverFraction', 0.9)
        self.mut_prob = self.moma.get('mutationRate', 0.1)
        self.n_proc = self.moma.get('numProcesses', 4)

# ---------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------
def load_environment(t, window_size):
    """
    Load background node metrics and network topology from Phase 1 CSVs.
    Returns averaged metrics for the specified time window.
    """
    metrics = pd.read_csv("phase1_cluster_metrics.csv")
    topo = pd.read_csv("phase1_network_topology.csv")
    
    # Slice data for the specific time window
    slice_df = metrics[(metrics['time'] >= t) & (metrics['time'] < t + window_size)]
    if slice_df.empty: raise ValueError("No data found for this time window")
    
    # Calculate average background load
    avg_load = slice_df.groupby(['node', 'device_type'])[['cpu_percent', 'memory_gb']].mean().reset_index()
    nodes = avg_load.to_dict('records')
    
    # Map node names to indices
    node_map = {n['node']: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)
    
    # Build Latency Matrix
    lat_mat = np.full((n_nodes, n_nodes), 60.0) # Default penalty for disconnected nodes
    np.fill_diagonal(lat_mat, 0.0)
    for _, row in topo.iterrows():
        if row['from_node'] in node_map and row['to_node'] in node_map:
            u, v = node_map[row['from_node']], node_map[row['to_node']]
            lat_mat[u, v] = row['latency_ms']
            
    return nodes, lat_mat, node_map

def load_services():
    """
    Load microservice requirements and interaction patterns from Phase 2 CSVs.
    """
    svc = pd.read_csv("phase2_microservices.csv")
    inter = pd.read_csv("phase2_service_interactions.csv")
    
    s_demands = {'cpu': svc['cpu_required'].values, 'mem': svc['memory_required'].values}
    
    interactions = []
    svc_map = {sid: i for i, sid in enumerate(svc['service_id'])}
    for _, row in inter.iterrows():
        if row['from_service'] in svc_map and row['to_service'] in svc_map:
            interactions.append({
                's1': svc_map[row['from_service']],
                's2': svc_map[row['to_service']],
                'cost': row['bandwidth_cost_mbps']
            })
    return svc, s_demands, interactions

# ---------------------------------------------------------
# Worker Initializer
# ---------------------------------------------------------
def init_worker(params):
    """
    Initialize static variables in worker processes for efficient parallel calculation.
    """
    MomaIndividual.num_nodes = params['n_nodes']
    MomaIndividual.num_services = params['n_services']
    MomaIndividual.latency_matrix = params['lat_mat']
    MomaIndividual.service_demands = params['s_demands']
    MomaIndividual.node_capacities = params['caps']
    MomaIndividual.node_bg_load = params['bg_load']
    MomaIndividual.interactions = params['interactions']

# ---------------------------------------------------------
# Smart Greedy Algorithm
# ---------------------------------------------------------
def generate_greedy(n_svc, n_nodes, caps, curr_cpu, curr_mem, s_demands):
    """
    Smart Greedy Allocation Logic:
    1. Strategy A (Best Fit): Try to find a node where the service FITS.
       If multiple nodes fit, choose the one with the most remaining resources (Balance).
    2. Strategy B (Least Bad): If NO node fits (all overloaded), find the node 
       that results in the MINIMUM overflow/violation.
    """
    assignment = [-1] * n_svc
    # Use temporary copies to track load during assignment
    temp_cpu = curr_cpu.copy()
    temp_mem = curr_mem.copy()
    
    for i in range(n_svc):
        best_n = -1
        
        # --- Strategy A: Find valid node with best balance ---
        max_score = -float('inf')
        feasible_found = False
        
        for n in range(n_nodes):
            rem_c = caps['cpu'][n] - temp_cpu[n]
            rem_m = caps['mem'][n] - temp_mem[n]
            
            # Check feasibility (Does it fit?)
            if rem_c >= s_demands['cpu'][i] and rem_m >= s_demands['mem'][i]:
                # Score: Normalized remaining capacity (Higher is better for load balancing)
                score = (rem_c / caps['cpu'][n]) + (rem_m / caps['mem'][n])
                if score > max_score:
                    max_score = score
                    best_n = n
                feasible_found = True
        
        # --- Strategy B: Minimize Overflow if Infeasible ---
        if not feasible_found:
            min_overflow = float('inf')
            best_n = -1 
            
            for n in range(n_nodes):
                # Calculate potential new load
                new_c = temp_cpu[n] + s_demands['cpu'][i]
                new_m = temp_mem[n] + s_demands['mem'][i]
                
                # Calculate overflow (Amount exceeding capacity)
                over_c = max(0, new_c - caps['cpu'][n])
                over_m = max(0, new_m - caps['mem'][n])
                
                # Normalize overflow by capacity to treat CPU/Mem equally
                overflow_score = (over_c / caps['cpu'][n]) + (over_m / caps['mem'][n])
                
                if overflow_score < min_overflow:
                    min_overflow = overflow_score
                    best_n = n
        
        # Assign to the selected node
        assignment[i] = best_n
        
        # Update load immediately for the next service
        target_n = assignment[i]
        temp_cpu[target_n] += s_demands['cpu'][i]
        temp_mem[target_n] += s_demands['mem'][i]
            
    return assignment

# ---------------------------------------------------------
# Local Evaluation Helpers
# ---------------------------------------------------------
def evaluate_locally(assignment, worker_params):
    """
    Helper function to calculate metrics for a single assignment.
    """
    init_worker(worker_params)
    ind = MomaIndividual(assignment)
    objs, cv = ind.calculate_objectives() # cv > 0 means Overloaded
    
    _, cur_cpu, cur_mem = ind.check_constraints()
    
    # Avoid division by zero
    caps_cpu = np.where(worker_params['caps']['cpu']==0, 1, worker_params['caps']['cpu'])
    caps_mem = np.where(worker_params['caps']['mem']==0, 1, worker_params['caps']['mem'])
    
    # Calculate Standard Deviation (Load Balance Metric)
    cpu_std = np.std(cur_cpu / caps_cpu)
    mem_std = np.std(cur_mem / caps_mem)
    
    return {
        'comm_cost': objs[0], 
        'cpu_std': cpu_std, 
        'mem_std': mem_std, 
        'is_feasible': (cv == 0), # True if NO overload
        'cv_amount': cv # Required for dynamic penalty calculation
    }

def get_node_details(assignment, worker_params, algo_name, t, idx_to_node_name):
    """
    Generate detailed node usage logs to PROVE constraints are checked.
    """
    # Initialize logic
    init_worker(worker_params)
    ind = MomaIndividual(assignment)
    # Get raw usage data (Background + Microservices)
    _, cur_cpu, cur_mem = ind.check_constraints()
    
    details = []
    for i in range(len(cur_cpu)):
        node_name = idx_to_node_name[i]
        cap_c = worker_params['caps']['cpu'][i]
        cap_m = worker_params['caps']['mem'][i]
        
        details.append({
            'Time_Window': t,
            'Algorithm': algo_name,
            'Node': node_name,
            # CPU Metrics
            'CPU_Used': round(cur_cpu[i], 2),
            'CPU_Limit': round(cap_c, 2),
            'CPU_Overloaded': cur_cpu[i] > cap_c + 0.001,
            # Memory Metrics
            'Mem_Used': round(cur_mem[i], 2),
            'Mem_Limit': round(cap_m, 2),
            'Mem_Overloaded': cur_mem[i] > cap_m + 0.001
        })
    return details

# ---------------------------------------------------------
# Main Comparison Loop
# ---------------------------------------------------------
def run_comparison(cfg):
    svc_df, s_demands, interactions = load_services()
    n_svc = len(svc_df)
    results = []
    detailed_allocations = [] 
    node_load_logs = [] 
    
    # [DYNAMIC PENALTY]
    # Penalize per unit of violation. 
    PENALTY_PER_UNIT = 6000.0 
    
    # Process every 5th time step
    for t in range(0, 100, 5): 
        try:
            print(f"Processing Window T={t}...")
            
            # Reset Random Seed
            current_seed = cfg.seed + t
            random.seed(current_seed)
            np.random.seed(current_seed)
            
            # 1. Load Environment
            nodes_data, lat_mat, node_map_dict = load_environment(t, 5)
            idx_to_node_name = {v: k for k, v in node_map_dict.items()}
            n_nodes = len(nodes_data)
            
            # Prepare Node Capacities & Background Load
            caps = {'cpu': [], 'mem': []}
            bg_load = {'cpu': [], 'mem': []}
            for n in nodes_data:
                spec = cfg.dev_specs[n['device_type']]
                caps['cpu'].append(spec['cpu_cores'])
                caps['mem'].append(spec['memory_gb'])
                bg_load['cpu'].append(spec['cpu_cores'] * n['cpu_percent'] / 100.0)
                bg_load['mem'].append(n['memory_gb'])
            
            for k in caps: caps[k] = np.array(caps[k])
            for k in bg_load: bg_load[k] = np.array(bg_load[k])

            worker_params = {
                'n_nodes': n_nodes, 'n_services': n_svc, 'lat_mat': lat_mat,
                's_demands': s_demands, 'caps': caps, 'bg_load': bg_load,
                'interactions': interactions
            }
            
            # ---------------------------------------------
            # Phase A: Smart Greedy (Baseline)
            # ---------------------------------------------
            greedy_assign = generate_greedy(n_svc, n_nodes, caps, bg_load['cpu'], bg_load['mem'], s_demands)
            greedy_metrics = evaluate_locally(greedy_assign, worker_params)
            node_load_logs.extend(get_node_details(greedy_assign, worker_params, 'Greedy', t, idx_to_node_name))
            
            # Apply Penalty if Overloaded
            greedy_final_cost = greedy_metrics['comm_cost']
            if not greedy_metrics['is_feasible']:
                violation = greedy_metrics['cv_amount']
                penalty = violation * PENALTY_PER_UNIT
                print(f"  [!] Greedy Overloaded ({violation:.2f})! Penalty: +{penalty:.0f}")
                greedy_final_cost += penalty
            
            # ---------------------------------------------
            # Phase B: MOMA
            # ---------------------------------------------
            pool = Pool(processes=cfg.n_proc, initializer=init_worker, initargs=(worker_params,))
            pop = MomaPopulation(0)
            
            # SEEDING
            pop.append(MomaIndividual(greedy_assign))
            # Fill remaining population
            for _ in range(cfg.pop_size - 1): pop.append(MomaIndividual())
            
            # Initial Evaluation
            pop.evaluateFitness(pool)
            pop.survivor_selection(cfg.pop_size)
            
            for gen in range(cfg.n_gen):
                offspring = pop.binary_tournament(cfg.pop_size)
                next_gen = MomaPopulation()
                
                # Crossover
                for i in range(0, cfg.pop_size, 2):
                    p1, p2 = offspring[i], offspring[i+1]
                    if random.random() < cfg.cx_prob:
                        c1, c2 = p1.crossover(p2)
                    else:
                        c1, c2 = MomaIndividual(p1.assignment), MomaIndividual(p2.assignment)
                    next_gen.append(c1); next_gen.append(c2)
                
                # Mutation
                for ind in next_gen: ind.mutate(cfg.mut_prob)
                
                # Evaluate & Selection
                next_gen.evaluateFitness(pool)
                pop.extend(next_gen)
                pop.survivor_selection(cfg.pop_size)
            
            pool.close(); pool.join()
            
            # ---------------------------------------------
            # Phase C: Select Best MOMA Solution
            # ---------------------------------------------
            valid_inds = [ind for ind in pop if ind.cv == 0]
            if not valid_inds: valid_inds = pop.population 
            
            fits = np.array([ind.objectives for ind in valid_inds])
            f_min, f_max = fits.min(axis=0), fits.max(axis=0)
            denom = f_max - f_min; denom[denom==0]=1.0
            f_norm = (fits - f_min) / denom
            
            # Weighted Selection
            weights = np.array([0.5, 0.5])
            best_idx = np.argmin(np.sum(f_norm * weights, axis=1))
            best_moma_assign = valid_inds[best_idx].assignment
            moma_metrics = evaluate_locally(best_moma_assign, worker_params)
            node_load_logs.extend(get_node_details(best_moma_assign, worker_params, 'MOMA', t, idx_to_node_name))
            
            # Apply Penalty if Overloaded
            moma_final_cost = moma_metrics['comm_cost']
            if not moma_metrics['is_feasible']:
                violation = moma_metrics['cv_amount']
                penalty = violation * PENALTY_PER_UNIT
                print(f"  [!] MOMA Overloaded ({violation:.2f})! Penalty: +{penalty:.0f}")
                moma_final_cost += penalty

            print(f"  Greedy Cost: {greedy_final_cost:.0f} | MOMA Cost: {moma_final_cost:.0f}")

            # --- Record Summary Metrics ---
            results.append({
                'time': t,
                'greedy_cpu_std': greedy_metrics['cpu_std'],
                'greedy_mem_std': greedy_metrics['mem_std'],
                'greedy_comm': greedy_final_cost,       # Penalized
                'moma_cpu_std': moma_metrics['cpu_std'],
                'moma_mem_std': moma_metrics['mem_std'],
                'moma_comm': moma_final_cost            # Penalized
            })

            # --- Record Detailed Allocation Log ---
            for i in range(n_svc):
                svc_name = svc_df.iloc[i]['service_name']
                
                g_node_idx = greedy_assign[i]
                g_node_name = idx_to_node_name[g_node_idx]
                
                m_node_idx = best_moma_assign[i]
                m_node_name = idx_to_node_name[m_node_idx]
                
                detailed_allocations.append({
                    'Time_Window': t,
                    'Service_ID': svc_df.iloc[i]['service_id'],
                    'Service_Name': svc_name,
                    'Greedy_Node': g_node_name,
                    'Greedy_Overloaded': not greedy_metrics['is_feasible'],
                    'MOMA_Node': m_node_name,
                    'MOMA_Overloaded': not moma_metrics['is_feasible']
                })

        except Exception as e:
            print(f"Error at T={t}: {e}")
            import traceback
            traceback.print_exc()
            break
            
    return pd.DataFrame(results), pd.DataFrame(detailed_allocations), pd.DataFrame(node_load_logs)

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
def plot_results(df):
    greedy_style = {'color': '#4682B4', 'marker': 'v', 'linestyle': '--', 'label': 'Greedy', 'alpha': 0.7}
    moma_style = {'color': '#DC143C', 'marker': 'o', 'linestyle': '-', 'label': 'MOMA'} 
    
    # 1. CPU Std
    plt.figure(figsize=(10,6))
    plt.plot(df['time'], df['greedy_cpu_std'], **greedy_style)
    plt.plot(df['time'], df['moma_cpu_std'], **moma_style)
    plt.title('(a) Computing Resource Balance Comparison')
    plt.xlabel('Time Window')
    plt.ylabel('Std Dev in CPU Utilization')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig("phase3_comparison_cpu.png")
    
    # 2. Mem Std
    plt.figure(figsize=(10,6))
    plt.plot(df['time'], df['greedy_mem_std'], **greedy_style)
    plt.plot(df['time'], df['moma_mem_std'], **moma_style)
    plt.title('(b) Memory Resource Balance Comparison')
    plt.xlabel('Time Window')
    plt.ylabel('Std Dev in Memory Utilization')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig("phase3_comparison_mem.png")
    
    # 3. Cost (Restored LaTeX)
    plt.figure(figsize=(10,6))
    plt.plot(df['time'], df['greedy_comm'], **greedy_style)
    plt.plot(df['time'], df['moma_comm'], **moma_style)
    plt.title('(c) Communication Cost Comparison')
    plt.xlabel('Time Window')
    plt.ylabel(r'Total Cost $\sum (Bandwidth Cost_{i,j} \times Latency_{node(i), node(j)})$')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig("phase3_comparison_cost.png")
    

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-i", "--input", dest="input", default="phase3_moma.cfg")
    (opts, args) = parser.parse_args()
    
    cfg = MomaConfig(opts.input)
    
    # Global Seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    res_df, detail_df, node_logs_df = run_comparison(cfg)
    
    res_df.to_csv("phase3_comparison_results.csv", index=False)
    detail_df.to_csv("phase3_detailed_allocations.csv", index=False)
    node_logs_df.to_csv("phase3_node_load_details.csv", index=False)
    
    print("  - phase3_comparison_results.csv (Summary)")
    print("  - phase3_detailed_allocations.csv (Allocation Plan)")
    print("  - phase3_node_load_details.csv (CPU & Memory Loads)")
    
    plot_results(res_df)