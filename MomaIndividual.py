import numpy as np
import random

class MomaIndividual:
    """
    Individual for MOMA Microservice Allocation (NSGA-II)
    Uses Integer Encoding: [Node_ID_for_Service_0, Node_ID_for_Service_1, ...]
    """
    # Static variables environment (set by init_worker)
    num_nodes = None
    num_services = None
    latency_matrix = None
    service_demands = None    # {'cpu': [], 'mem': []}
    node_capacities = None    # {'cpu': [], 'mem': []}
    node_bg_load = None       # {'cpu': [], 'mem': []}
    interactions = None       # List of dicts

    def __init__(self, assignment=None):
        if assignment is None:
            # Random initialization
            self.assignment = np.random.randint(0, self.num_nodes, self.num_services)
        else:
            # Seeded initialization
            self.assignment = np.array(assignment, dtype=int)
            
        self.objectives = None # [comm_cost, balance_std]
        self.cv = None         # Constraint Violation
        self.rank = None       # Pareto Rank
        self.crowding_dist = 0 # Crowding Distance
        
    def check_constraints(self):
        """Calculate Constraint Violation (CV)"""
        current_cpu = self.node_bg_load['cpu'].copy()
        current_mem = self.node_bg_load['mem'].copy()
        
        # Add service load
        for s_idx, n_idx in enumerate(self.assignment):
            current_cpu[n_idx] += self.service_demands['cpu'][s_idx]
            current_mem[n_idx] += self.service_demands['mem'][s_idx]
            
        # Calculate overload (CV)
        cv_cpu = np.maximum(0, current_cpu - self.node_capacities['cpu'])
        cv_mem = np.maximum(0, current_mem - self.node_capacities['mem'])
        total_cv = np.sum(cv_cpu) + np.sum(cv_mem)
        
        return total_cv, current_cpu, current_mem

    def calculate_objectives(self):
        """Calculate Cost and Balance"""
        # 1. Constraints & Load
        cv, cur_cpu, cur_mem = self.check_constraints()
        self.cv = cv
        
        # 2. Communication Cost (Obj 1)
        comm_cost = 0.0
        for inter in self.interactions:
            n1 = self.assignment[inter['s1']]
            n2 = self.assignment[inter['s2']]
            if n1 != n2:
                comm_cost += inter['cost'] * self.latency_matrix[n1, n2]
                
        # 3. Load Balance Std Dev (Obj 2)
        # Avoid division by zero
        caps_cpu = np.where(self.node_capacities['cpu']==0, 1, self.node_capacities['cpu'])
        caps_mem = np.where(self.node_capacities['mem']==0, 1, self.node_capacities['mem'])
        
        cpu_util = cur_cpu / caps_cpu
        mem_util = cur_mem / caps_mem
        
        balance_score = (np.std(cpu_util) + np.std(mem_util)) / 2.0
        
        self.objectives = [comm_cost, balance_score]
        return self.objectives, self.cv

    def crossover(self, other):
        """Uniform Crossover"""
        mask = np.random.rand(self.num_services) < 0.5
        child1_assign = np.where(mask, self.assignment, other.assignment)
        child2_assign = np.where(mask, other.assignment, self.assignment)
        return MomaIndividual(child1_assign), MomaIndividual(child2_assign)
    
    def mutate(self, mutation_rate):
        """Swap Mutation"""
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(self.num_services), 2)
            self.assignment[idx1], self.assignment[idx2] = \
                self.assignment[idx2], self.assignment[idx1]
            # Reset metrics
            self.objectives = None
            self.cv = None
            self.rank = None

# Worker class for parallel fitness evaluation
class Worker:
    @classmethod    
    def evaluateFitness(cls, assignment):
        ind = MomaIndividual(assignment)
        objs, cv = ind.calculate_objectives()
        return objs, cv