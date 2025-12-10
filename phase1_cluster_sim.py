# To run: python phase1_cluster_sim.py -i device_spec.cfg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optparse
import sys
import yaml

# -----------------------------
# Configuration Class
# -----------------------------
class ClusterConfig:
    """
    Configuration class for Cluster Simulation
    """
    options = {
        'Simulation': {
            'randomSeed': (int, True),
            'simulationSteps': (int, True)
        },
        'DeviceSpecs': (dict, True)
    }

    def __init__(self, inFileName):
        with open(inFileName, 'r') as infile:
            ymlcfg = yaml.safe_load(infile)

        # Process Simulation Section
        sim_cfg = ymlcfg.get('Simulation', None)
        if sim_cfg is None:
            raise Exception('Missing "Simulation" section in cfg file')
        
        for opt, (opt_type, mandatory) in self.options['Simulation'].items():
            if opt in sim_cfg:
                val = sim_cfg[opt]
                if type(val) != opt_type:
                    raise Exception(f'Parameter "{opt}" has wrong type')
                setattr(self, opt, val)
        
        # Process DeviceSpecs Section
        dev_cfg = ymlcfg.get('DeviceSpecs', None)
        if dev_cfg is None:
            raise Exception('Missing "DeviceSpecs" section in cfg file')
        self.deviceSpecs = dev_cfg

# -----------------------------
# Heterogeneous Node Simulator
# -----------------------------
class HeterogeneousNode:
    def __init__(self, cluster_id, node_id, device_type, specs):
        """
        Simulate a heterogeneous computing node
        """
        self.cluster_id = cluster_id
        self.node_id = node_id
        self.device_type = device_type
        self.name = f"C{cluster_id}_N{node_id}_{device_type[:2]}"
        
        # Load specs from Config
        self.cpu_cores = specs['cpu_cores']
        self.total_memory_gb = specs['memory_gb']
        self.max_bandwidth_mbps = specs['max_bandwidth_mbps']
        self.power_watts = specs['power_watts']
        
        # Initialize resource states
        if device_type == 'PC':
            self.cpu_prev = np.random.uniform(10, 25)
            self.mem_prev = np.random.uniform(2, 4)
            self.bw_prev = np.random.uniform(50, 100)
        elif device_type == 'RaspberryPi':
            self.cpu_prev = np.random.uniform(30, 50)
            self.mem_prev = np.random.uniform(1, 2)
            self.bw_prev = np.random.uniform(15, 30)
        else:  # JetsonNano
            self.cpu_prev = np.random.uniform(20, 35)
            self.mem_prev = np.random.uniform(1.5, 2.5)
            self.bw_prev = np.random.uniform(40, 80)
        
        self.phase = 'medium'
        self.phase_counter = 0
    
    def simulate_cpu(self):
        """CPU usage simulation"""
        if self.device_type == 'PC':
            baselines = {'low': 15, 'medium': 30, 'high': 55}
            noise_std = 5
        elif self.device_type == 'RaspberryPi':
            baselines = {'low': 40, 'medium': 60, 'high': 80}
            noise_std = 8
        else:  # JetsonNano
            baselines = {'low': 25, 'medium': 45, 'high': 65}
            noise_std = 6
        
        baseline = baselines[self.phase]
        alpha = 0.8
        noise = np.random.normal(0, noise_std)
        cpu = alpha * self.cpu_prev + (1 - alpha) * baseline + noise
        
        if np.random.rand() < 0.05:
            cpu += np.random.uniform(10, 20)
        
        max_cpu = 95 if self.device_type == 'PC' else 90
        cpu = np.clip(cpu, 5, max_cpu)
        self.cpu_prev = cpu
        return round(cpu, 2)
    
    def simulate_memory(self):
        """Memory usage simulation"""
        drift = 0.015 if self.device_type == 'PC' else 0.01
        noise = np.random.normal(0, 0.08)
        mem = self.mem_prev + drift + noise
        
        gc_prob = 0.08 if self.device_type != 'PC' else 0.05
        if np.random.rand() < gc_prob:
            mem *= np.random.uniform(0.5, 0.7)
        
        if mem > self.total_memory_gb * 0.9:
            mem = self.total_memory_gb * 0.6
        
        mem = np.clip(mem, 0.5, self.total_memory_gb * 0.95)
        self.mem_prev = mem
        return round(mem, 2)
    
    def simulate_bandwidth(self):
        """Bandwidth usage simulation"""
        burst_prob = 0.15 if self.phase == 'high' else 0.08
        
        if np.random.rand() < burst_prob:
            if self.device_type == 'PC':
                bw = np.random.uniform(200, 300)
            elif self.device_type == 'RaspberryPi':
                bw = np.random.uniform(50, 70)
            else:  # JetsonNano
                bw = np.random.uniform(150, 250)
        else:
            if self.device_type == 'PC':
                base = np.random.uniform(50, 150)
                jitter = np.random.uniform(-20, 20)
            elif self.device_type == 'RaspberryPi':
                base = np.random.uniform(15, 40)
                jitter = np.random.uniform(-5, 5)
            else:  # JetsonNano
                base = np.random.uniform(40, 120)
                jitter = np.random.uniform(-15, 15)
            
            bw = self.bw_prev * 0.7 + base * 0.3 + jitter
        
        bw = np.clip(bw, 5, self.max_bandwidth_mbps)
        self.bw_prev = bw
        return round(bw, 2)
    
    def update_phase(self):
        """Update workload phase"""
        self.phase_counter += 1
        if self.phase_counter > np.random.randint(40, 80):
            self.phase_counter = 0
            self.phase = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
    
    def step(self):
        """Execute one simulation timestep"""
        self.update_phase()
        
        # Only return DYNAMIC usage metrics.
        return {
            'cluster': self.cluster_id,
            'node': self.name,
            'device_type': self.device_type,
            'cpu_percent': self.simulate_cpu(),
            'memory_gb': self.simulate_memory(),
            'bandwidth_mbps': self.simulate_bandwidth()
        }

# -----------------------------
# Simulate Clusters
# -----------------------------
def simulate_heterogeneous_clusters(cfg):
    """Simulate clusters using Config"""
    print("Starting heterogeneous hybrid cluster simulation...")
    
    s = cfg.deviceSpecs
    nodes = [
        HeterogeneousNode(1, 1, 'PC', s['PC']),
        HeterogeneousNode(1, 2, 'RaspberryPi', s['RaspberryPi']),
        HeterogeneousNode(2, 1, 'PC', s['PC']),
        HeterogeneousNode(2, 2, 'RaspberryPi', s['RaspberryPi']),
        HeterogeneousNode(3, 1, 'PC', s['PC']),
        HeterogeneousNode(3, 2, 'JetsonNano', s['JetsonNano'])
    ]
    
    records = []
    for t in range(cfg.simulationSteps):
        for node in nodes:
            metrics = node.step()
            metrics['time'] = t
            records.append(metrics)
    
    df = pd.DataFrame(records)

    # Reordering columns for clarity
    df = df[['time', 'cluster', 'node', 'device_type', 
             'cpu_percent', 'memory_gb', 'bandwidth_mbps']]
    
    return df, nodes

def generate_network_topology(nodes):
    """Generate latency (Original Logic)"""
    network_data = []
    device_tier = {'PC': 0, 'JetsonNano': 1, 'RaspberryPi': 2}
    
    for i, node_from in enumerate(nodes):
        for j, node_to in enumerate(nodes):
            if i != j:
                from_device = node_from.device_type
                to_device = node_to.device_type
                combined_tier = device_tier[from_device] + device_tier[to_device]
                
                if node_from.cluster_id == node_to.cluster_id:
                    if combined_tier == 0: latency = np.random.uniform(0.3, 0.6)
                    elif combined_tier == 1: latency = np.random.uniform(0.6, 0.9)
                    elif combined_tier == 2:
                        if from_device == 'JetsonNano' and to_device == 'JetsonNano': latency = np.random.uniform(0.9, 1.2)
                        else: latency = np.random.uniform(1.2, 1.5)
                    elif combined_tier == 3: latency = np.random.uniform(1.5, 1.8)
                    else: latency = np.random.uniform(1.8, 2.1)
                else:
                    if combined_tier == 0: latency = np.random.uniform(50, 55)
                    elif combined_tier == 1: latency = np.random.uniform(55, 60)
                    elif combined_tier == 2:
                        if from_device == 'JetsonNano' and to_device == 'JetsonNano': latency = np.random.uniform(60, 65)
                        else: latency = np.random.uniform(65, 70)
                    elif combined_tier == 3: latency = np.random.uniform(70, 75)
                    else: latency = np.random.uniform(75, 80)
                
                network_data.append({
                    'from_node': node_from.name,
                    'to_node': node_to.name,
                    'from_cluster': node_from.cluster_id,
                    'to_cluster': node_to.cluster_id,
                    'latency_ms': round(latency, 2)
                })
    
    return pd.DataFrame(network_data)

# -----------------------------
# Analysis & Visualization
# -----------------------------
def analyze_cluster_stats(df, nodes):
    """
    Print cluster statistics.
    [CRITICAL CHANGE] Reconstructs the print output by looking up specs from
    the 'nodes' list, since those columns are no longer in the DataFrame.
    """
    print("\n" + "="*80)
    print("HETEROGENEOUS CLUSTER STATISTICS")
    print("="*80)
    
    # Map node name to node object to retrieve static specs
    node_map = {n.name: n for n in nodes}
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        print(f"\n[Cluster {cluster_id}]")
        
        for node_name in cluster_data['node'].unique():
            node_data = cluster_data[cluster_data['node'] == node_name]
            
            # Retrieve static specs from the node object
            node_obj = node_map[node_name]
            specs = f"{node_obj.cpu_cores}C/{node_obj.total_memory_gb}GB/{node_obj.max_bandwidth_mbps}Mbps"
            
            print(f"\n  {node_name} ({node_obj.device_type}, {specs}):")
            print(f"    CPU:       Avg {node_data['cpu_percent'].mean():.1f}% | Max {node_data['cpu_percent'].max():.1f}%")
            print(f"    Memory:    Avg {node_data['memory_gb'].mean():.2f} GB | Max {node_data['memory_gb'].max():.2f} GB")
            print(f"    Bandwidth: Avg {node_data['bandwidth_mbps'].mean():.1f} Mbps | Max {node_data['bandwidth_mbps'].max():.1f} Mbps")
    
    print("\n" + "="*80)

def plot_heterogeneous_resources(df, save_path='phase1_heterogeneous_clusters.png'):
    """Visualize resource usage (Original)"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    nodes = df['node'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(nodes)))
    
    # CPU Usage
    for idx, node in enumerate(nodes):
        node_data = df[df['node'] == node]
        device = node_data['device_type'].iloc[0]
        label = f"{node} ({device})"
        axes[0].plot(node_data['time'], node_data['cpu_percent'], 
                     label=label, color=colors[idx], alpha=0.8, linewidth=1.2)
    axes[0].set_ylabel('CPU Usage (%)', fontsize=12)
    axes[0].set_title('Heterogeneous Hybrid Cluster - Resource Monitoring (Simulated Grafana Data)', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', ncol=2, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # Memory Usage
    for idx, node in enumerate(nodes):
        node_data = df[df['node'] == node]
        axes[1].plot(node_data['time'], node_data['memory_gb'], 
                     label=node, color=colors[idx], alpha=0.8, linewidth=1.2)
    axes[1].set_ylabel('Memory Usage (GB)', fontsize=12)
    axes[1].legend(loc='upper right', ncol=2, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    # Bandwidth Usage
    for idx, node in enumerate(nodes):
        node_data = df[df['node'] == node]
        axes[2].plot(node_data['time'], node_data['bandwidth_mbps'], 
                     label=node, color=colors[idx], alpha=0.8, linewidth=1.2)
    axes[2].set_xlabel('Time Step', fontsize=12)
    axes[2].set_ylabel('Bandwidth (Mbps)', fontsize=12)
    axes[2].legend(loc='upper right', ncol=2, fontsize=9)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {save_path}")
    plt.show()

# -----------------------------
# Main
# -----------------------------
def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    try:
        # 1. Parse Args
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputFileName", 
                          help="input configuration file", default=None)
        (options, args) = parser.parse_args(argv)
        
        if options.inputFileName is None:
            raise Exception("Must specify input file name using -i or --input option.")
            
        # 2. Load Config
        cfg = ClusterConfig(options.inputFileName)
        np.random.seed(cfg.randomSeed)
        
        # 3. Print Header
        print("="*80)
        print("PHASE 1: Heterogeneous Hybrid Cluster Traffic Simulation")
        print("         (Simulating Grafana Monitoring Data)")
        print("="*80)
        
        # 4. Simulate
        df, nodes = simulate_heterogeneous_clusters(cfg)
        network_df = generate_network_topology(nodes)
        
        # 5. Save CSV
        df.to_csv('phase1_cluster_metrics.csv', index=False)
        network_df.to_csv('phase1_network_topology.csv', index=False)
        
        print(f"\n✓ Saved: phase1_cluster_metrics.csv ({len(df)} rows)")
        print(f"✓ Saved: phase1_network_topology.csv ({len(network_df)} rows)")
        
        # 6. Preview
        print("\n[cluster_metrics.csv Preview]")
        print(df.head(10))
        
        print("\n[network_topology.csv Preview]")
        print(network_df.head(8))
        
        # 7. Statistics
        analyze_cluster_stats(df, nodes)
        
        # 8. Visualize
        print("\nGenerating visualization...")
        plot_heterogeneous_resources(df)
        
        # 9. Footer
        print("\n" + "="*80)
        print("PHASE 1 COMPLETED!")
        print("="*80)
        print("\nOutput Files:")
        print("  1. phase1_cluster_metrics.csv       - Heterogeneous node resource usage data")
        print("  2. phase1_network_topology.csv      - Inter-node network topology")
        print("  3. phase1_heterogeneous_clusters.png - Resource usage trend chart")
        print("\nKey Concepts:")
        print("  • CPU:       Processor utilization (%), reflects computing load")
        print("  • Memory:    Memory usage (GB), reflects data storage demand")
        print("  • Bandwidth: Network traffic (Mbps), includes container communication, data sync, etc.")
        print("  • Baseline:  Target CPU value, automatically adjusted by workload phase")
        
        # Recover specs for print output
        print(f"\nDevice Specifications (Loaded from {options.inputFileName}):")
        for dev, spec in cfg.deviceSpecs.items():
             print(f"  • {dev:<14}: {spec['cpu_cores']} cores / {spec['memory_gb']} GB / {spec['max_bandwidth_mbps']} Mbps")
        
        print("\nBandwidth Usage Targets:")
        print("  • PC:            Normal 5-15% (50-150 Mbps), Burst 20-30% (200-300 Mbps)")
        print("  • Raspberry Pi:  Normal 15-40% (15-40 Mbps), Burst 50-70% (50-70 Mbps)")
        print("  • Jetson Nano:   Normal 5-16% (40-120 Mbps), Burst 20-33% (150-250 Mbps)")
        print("\nNext Step: Phase 2 - Define Microservices and their resource requirements")
        print("="*80)
        
    except Exception as info:
        print(f"Error: {info}")
        sys.exit(1)

if __name__ == "__main__":
    main()