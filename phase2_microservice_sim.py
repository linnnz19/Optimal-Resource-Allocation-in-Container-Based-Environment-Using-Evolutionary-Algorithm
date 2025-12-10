# To run: python phase2_microservice_sim.py -i device_spec.cfg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import optparse
import sys
import yaml

# -----------------------------
# Configuration Class
# -----------------------------
class MicroserviceConfig:
    """
    Configuration class for Microservice Simulation
    Reads simple simulation settings (Seed, Count)
    """
    options = {
        'Simulation': {
            'randomSeed': (int, True),
            'microserviceCount': (int, True)
        }
    }

    def __init__(self, inFileName):
        with open(inFileName, 'r') as infile:
            ymlcfg = yaml.safe_load(infile)

        sim_cfg = ymlcfg.get('Simulation', None)
        if sim_cfg:
            self.randomSeed = sim_cfg.get('randomSeed', 42)
            self.count = sim_cfg.get('microserviceCount', 10)
        else:
            raise Exception("Missing 'Simulation' section in CFG")

# -----------------------------
# Microservice Definition
# -----------------------------
class Microservice:
    def __init__(self, service_id, name, service_type):
        """
        Define a microservice with resource requirements
        """
        self.service_id = service_id
        self.name = name
        self.service_type = service_type
        
        # Define resource requirements based on service type
        self.cpu_request, self.memory_request, self.bandwidth_request = self._define_resources()
    
    def _define_resources(self):
        """
        Define resource requirements based on service type (Randomized)
        Returns: (cpu_cores, memory_gb, bandwidth_mbps)
        """
        resource_profiles = {
            'frontend': {
                'cpu': np.random.uniform(0.5, 1.5),
                'memory': np.random.uniform(0.5, 1.5),
                'bandwidth': np.random.uniform(20, 50)
            },
            'backend': {
                'cpu': np.random.uniform(1.0, 3.0),
                'memory': np.random.uniform(1.0, 3.0),
                'bandwidth': np.random.uniform(30, 80)
            },
            'database': {
                'cpu': np.random.uniform(0.5, 2.0),
                'memory': np.random.uniform(3.0, 6.0),
                'bandwidth': np.random.uniform(40, 100)
            },
            'cache': {
                'cpu': np.random.uniform(0.3, 1.0),
                'memory': np.random.uniform(1.5, 3.0),
                'bandwidth': np.random.uniform(50, 150)
            },
            'ml_inference': {
                'cpu': np.random.uniform(2.0, 4.0),
                'memory': np.random.uniform(2.0, 4.0),
                'bandwidth': np.random.uniform(20, 60)
            },
            'message_queue': {
                'cpu': np.random.uniform(0.5, 1.5),
                'memory': np.random.uniform(1.0, 2.5),
                'bandwidth': np.random.uniform(40, 120)
            }
        }
        
        profile = resource_profiles[self.service_type]
        return (
            round(profile['cpu'], 2),
            round(profile['memory'], 2),
            round(profile['bandwidth'], 2)
        )
    
    def to_dict(self):
        """Convert to dictionary for DataFrame"""
        return {
            'service_id': self.service_id,
            'service_name': self.name,
            'service_type': self.service_type,
            'cpu_required': self.cpu_request,
            'memory_required': self.memory_request,
            'bandwidth_required': self.bandwidth_request
        }

# -----------------------------
# Generate Microservices
# -----------------------------
def generate_microservices(num_services=10):
    """Generate microservices for a typical web application"""
    print(f"Generating {num_services} microservices...")
    
    services = []
    # Base list of services, if num_services > 10, logic can be extended or cycled
    service_configs = [
        ('ms_1', 'Web Frontend', 'frontend'),
        ('ms_2', 'Mobile API Gateway', 'frontend'),
        ('ms_3', 'User Service', 'backend'),
        ('ms_4', 'Order Service', 'backend'),
        ('ms_5', 'Payment Service', 'backend'),
        ('ms_6', 'Inventory Service', 'backend'),
        ('ms_7', 'PostgreSQL Database', 'database'),
        ('ms_8', 'Redis Cache', 'cache'),
        ('ms_9', 'Recommendation Engine', 'ml_inference'),
        ('ms_10', 'RabbitMQ Message Queue', 'message_queue')
    ]
    
    for i in range(min(num_services, len(service_configs))):
        sid, name, stype = service_configs[i]
        services.append(Microservice(sid, name, stype))
    
    return services

# -----------------------------
# Define Service Interactions
# -----------------------------
def generate_service_interactions(services):
    """Generate communication patterns (Original Logic)"""
    print("\nGenerating service interaction matrix...")
    
    n = len(services)
    interaction_matrix = np.zeros((n, n))
    
    for i, from_service in enumerate(services):
        for j, to_service in enumerate(services):
            if i == j: continue
            
            from_type = from_service.service_type
            to_type = to_service.service_type
            
            if from_type == 'frontend' and to_type == 'backend':
                interaction_matrix[i][j] = np.random.randint(7, 11)
            elif from_type == 'backend' and to_type == 'database':
                interaction_matrix[i][j] = np.random.randint(8, 11)
            elif from_type == 'backend' and to_type == 'cache':
                interaction_matrix[i][j] = np.random.randint(7, 11)
            elif from_type == 'backend' and to_type == 'ml_inference':
                interaction_matrix[i][j] = np.random.randint(4, 7)
            elif from_type == 'backend' and to_type == 'message_queue':
                interaction_matrix[i][j] = np.random.randint(5, 8)
            elif from_type == 'backend' and to_type == 'backend':
                interaction_matrix[i][j] = np.random.randint(2, 6)
            elif from_type == 'frontend' and to_type == 'cache':
                interaction_matrix[i][j] = np.random.randint(4, 7)
    
    interaction_records = []
    # Calculate bandwidth cost based on frequency and average bandwidth requirement
    for i, from_service in enumerate(services):
        for j, to_service in enumerate(services):
            if interaction_matrix[i][j] > 0:
                frequency = int(interaction_matrix[i][j])
                avg_bandwidth = (from_service.bandwidth_request + to_service.bandwidth_request) / 2
                bandwidth_cost = round((frequency / 10.0) * avg_bandwidth, 2)
                
                interaction_records.append({
                    'from_service': from_service.service_id,
                    'to_service': to_service.service_id,
                    'communication_frequency': frequency,
                    'bandwidth_cost_mbps': bandwidth_cost
                })
    
    interaction_df = pd.DataFrame(interaction_records)
    return interaction_df, interaction_matrix

# -----------------------------
# Visualization
# -----------------------------
def visualize_microservices(services_df, save_path='phase2_microservices.png'):
    """Visualize microservice resource requirements"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    service_types = services_df['service_type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(service_types)))
    color_map = {stype: colors[i] for i, stype in enumerate(service_types)}
    service_colors = [color_map[stype] for stype in services_df['service_type']]
    
    axes[0].barh(services_df['service_id'], services_df['cpu_required'], color=service_colors)
    axes[0].set_xlabel('CPU Cores Required', fontsize=11)
    axes[0].set_title('CPU Requirements', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    axes[1].barh(services_df['service_id'], services_df['memory_required'], color=service_colors)
    axes[1].set_xlabel('Memory (GB) Required', fontsize=11)
    axes[1].set_title('Memory Requirements', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    axes[2].barh(services_df['service_id'], services_df['bandwidth_required'], color=service_colors)
    axes[2].set_xlabel('Bandwidth (Mbps) Required', fontsize=11)
    axes[2].set_title('Bandwidth Requirements', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='x')
    
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[stype], label=stype) 
                      for stype in service_types]
    fig.legend(handles=legend_elements, loc='upper center', ncol=len(service_types), 
              bbox_to_anchor=(0.5, 0.98), fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {save_path}")
    plt.show()

def visualize_service_interactions(services, interaction_df, save_path='phase2_interactions.png'):
    """Visualize microservice interaction graph"""
    fig, ax = plt.subplots(figsize=(12, 9))
    G = nx.DiGraph()
    
    for service in services:
        G.add_node(service.service_id, service_type=service.service_type)
    
    for _, row in interaction_df.iterrows():
        G.add_edge(row['from_service'], row['to_service'], 
                  frequency=row['communication_frequency'],
                  bandwidth=row['bandwidth_cost_mbps'])
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    service_type_colors = {
        'frontend': '#FF6B6B', 'backend': '#4ECDC4', 'database': '#FFE66D',
        'cache': '#95E1D3', 'ml_inference': '#C7CEEA', 'message_queue': '#FFDAB9'
    }
    node_colors = [service_type_colors.get(G.nodes[node]['service_type'], '#CCCCCC') for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, alpha=0.9, ax=ax)
    labels = {node: f"{node}\n{G.nodes[node]['service_type']}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold', ax=ax)
    
    edges = G.edges()
    bandwidths = [G[u][v]['bandwidth'] for u, v in edges]
    max_bandwidth = max(bandwidths) if bandwidths else 1
    edge_widths = [(bw / max_bandwidth) * 5 for bw in bandwidths]
    
    frequencies = [G[u][v]['frequency'] for u, v in edges]
    edge_colors = []
    for f in frequencies:
        if f >= 7: edge_colors.append('#E74C3C')
        elif f >= 4: edge_colors.append('#F39C12')
        else: edge_colors.append('#95A5A6')
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color=edge_colors, arrows=True, 
                          arrowsize=15, arrowstyle='->', ax=ax)
    
    ax.set_title('Microservice Communication Graph\n(Edge thickness = Bandwidth cost, Color = Frequency)', 
                fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='High Frequency (7-10)'),
        Patch(facecolor='#F39C12', label='Medium Frequency (4-6)'),
        Patch(facecolor='#95A5A6', label='Low Frequency (1-3)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path}")
    plt.show()

def print_summary(services_df, interaction_df):
    """Print summary statistics (Original)"""
    print("\n" + "="*80)
    print("MICROSERVICE SUMMARY")
    print("="*80)
    
    print("\n[Resource Requirements]")
    print(f"  Total CPU Required:       {services_df['cpu_required'].sum():.2f} cores")
    print(f"  Total Memory Required:    {services_df['memory_required'].sum():.2f} GB")
    print(f"  Total Bandwidth Required: {services_df['bandwidth_required'].sum():.2f} Mbps")
    
    print("\n[Service Type Distribution]")
    type_counts = services_df['service_type'].value_counts()
    for stype, count in type_counts.items():
        print(f"  {stype}: {count} service(s)")
    
    print("\n[Communication Statistics]")
    print(f"  Total communication links: {len(interaction_df)}")
    print(f"  Total bandwidth cost:      {interaction_df['bandwidth_cost_mbps'].sum():.2f} Mbps")
    
    high_freq = len(interaction_df[interaction_df['communication_frequency'] >= 7])
    med_freq = len(interaction_df[(interaction_df['communication_frequency'] >= 4) & 
                                   (interaction_df['communication_frequency'] < 7)])
    low_freq = len(interaction_df[interaction_df['communication_frequency'] < 4])
    
    print(f"  High frequency (7-10):     {high_freq} links")
    print(f"  Medium frequency (4-6):    {med_freq} links")
    print(f"  Low frequency (1-3):       {low_freq} links")
    
    print("\n[Top 5 Highest Bandwidth Communications]")
    top_comm = interaction_df.nlargest(5, 'bandwidth_cost_mbps')
    for _, row in top_comm.iterrows():
        print(f"  {row['from_service']} → {row['to_service']}: "
              f"{row['bandwidth_cost_mbps']:.2f} Mbps (freq={row['communication_frequency']})")
    
    print("\n" + "="*80)

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
        cfg = MicroserviceConfig(options.inputFileName)
        np.random.seed(cfg.randomSeed)
        
        # 3. Print Header
        print("="*80)
        print("PHASE 2: Microservice Definition & Communication Patterns")
        print("="*80)
        
        # 4. Generate Data
        services = generate_microservices(num_services=cfg.count)
        services_df = pd.DataFrame([s.to_dict() for s in services])
        
        interaction_df, interaction_matrix = generate_service_interactions(services)
        
        # 5. Save CSV
        services_df.to_csv('phase2_microservices.csv', index=False)
        interaction_df.to_csv('phase2_service_interactions.csv', index=False)
        
        print(f"\n✓ Saved: phase2_microservices.csv ({len(services_df)} services)")
        print(f"✓ Saved: phase2_service_interactions.csv ({len(interaction_df)} communication links)")
        
        # 6. Preview
        print("\n[Microservice Information Preview]")
        print(services_df)
        
        print("\n[Service Communication Preview (first 10 rows)]")
        print(interaction_df.head(10))
        
        # 7. Statistics
        print_summary(services_df, interaction_df)
        
        # 8. Visualize
        print("\nGenerating visualizations...")
        visualize_microservices(services_df)
        visualize_service_interactions(services, interaction_df)
        
        # 9. Footer
        print("\n" + "="*80)
        print("PHASE 2 COMPLETED!")
        print("="*80)
        print("\nOutput Files:")
        print("  1. phase2_microservices.csv         - Service resource requirements")
        print("  2. phase2_service_interactions.csv  - Communication patterns")
        print("  3. phase2_microservices.png         - Resource requirements chart")
        print("  4. phase2_interactions.png          - Communication graph")
        print("\n" + "="*80)
        print("KEY CONCEPTS FOR NSGA-II:")
        print("="*80)
        print("\n[Service-Level Resources (from phase2_microservices.csv)]")
        print("  • cpu_required:       CPU cores needed by each service")
        print("  • memory_required:    Memory (GB) needed by each service")
        print("  • bandwidth_required: Baseline network traffic generated by service")
        print("\n[Node-Level Resources (from phase1_cluster_metrics.csv)]")
        print("  • cpu_cores:          Total CPU capacity of node")
        print("  • total_memory_gb:    Total memory capacity of node")
        print("  • max_bandwidth_mbps: Total network capacity of node")
        print("\n[Communication Cost (from phase2_service_interactions.csv)]")
        print("  • communication_frequency:  How often services communicate (1-10)")
        print("  • bandwidth_cost_mbps:      Actual bandwidth consumed by communication")
        print("\n[NSGA-II Objective Functions]")
        print("  Objective 1 - Minimize Communication Cost:")
        print("    Sum of (frequency × latency) for all service pairs on different nodes")
        print("    → Services that talk frequently should be placed close together")
        print("\n  Objective 2 - Maximize Resource Utilization:")
        print("    Balance CPU/Memory/Bandwidth usage across all nodes")
        print("    → Avoid overloading some nodes while others are idle")
        print("\n[Constraints]")
        print("  • CPU:       Σ(service.cpu_required) ≤ node.cpu_cores")
        print("  • Memory:    Σ(service.memory_required) ≤ node.total_memory_gb")
        print("  • Bandwidth: Σ(service.bandwidth_required + communication_cost) ≤ node.max_bandwidth_mbps")
        print("\nNext Step: Phase 3 - Implement MOMA (NSGA-II) algorithm")
        print("="*80)
        
    except Exception as info:
        print(f"Error: {info}")
        sys.exit(1)

if __name__ == "__main__":
    main()