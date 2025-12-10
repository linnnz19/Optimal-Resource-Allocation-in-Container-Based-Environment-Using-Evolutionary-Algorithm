import copy
import random
import numpy as np
from operator import attrgetter
from MomaIndividual import MomaIndividual, Worker

class MomaPopulation:
    def __init__(self, populationSize=0):
        self.population = []
        for _ in range(populationSize):
            self.population.append(MomaIndividual())
            
    def __len__(self):
        return len(self.population)
    
    def __getitem__(self, key):
        return self.population[key]
        
    def append(self, ind):
        self.population.append(ind)
        
    def extend(self, other_pop):
        self.population.extend(other_pop.population)

    def evaluateFitness(self, pool=None):
        """Evaluate fitness in parallel with chunksize optimization"""
        assignments = [ind.assignment for ind in self.population]
        
        if pool:
            n_tasks = len(assignments)
            n_proc = pool._processes
            chunk = max(1, n_tasks // (n_proc * 4))
            results = pool.map(Worker.evaluateFitness, assignments, chunksize=chunk)
        else:
            results = [Worker.evaluateFitness(a) for a in assignments]
            
        for i, (objs, cv) in enumerate(results):
            self.population[i].objectives = objs
            self.population[i].cv = cv

    # --------------------------------------------
    # [FAST] Vectorized Non-Dominated Sort
    # --------------------------------------------
    def calc_domination_rank(self):
        pop_size = len(self.population)
        if pop_size == 0: return []
        
        objs = np.array([ind.objectives for ind in self.population])
        cvs = np.array([ind.cv for ind in self.population])
        
        # Dominance Matrix Logic
        # 1. Feasibility: Lower CV dominates
        dom_cv = (cvs[:, None] < cvs) | ((cvs[:, None] == 0) & (cvs > 0))
        
        # 2. Objectives: Both feasible, Lower Objs dominate
        feasible_mask = (cvs[:, None] == 0) & (cvs == 0)
        better_or_equal = (objs[:, None] <= objs).all(axis=2)
        better = (objs[:, None] < objs).any(axis=2)
        dom_obj = feasible_mask & better_or_equal & better
        
        dominates = dom_cv | dom_obj
        
        # Rank Calculation
        domination_counts = dominates.sum(axis=0)
        dominated_sets = [np.where(dominates[i])[0] for i in range(pop_size)]
        
        fronts = [[]]
        for i in range(pop_size):
            self.population[i].domination_count = domination_counts[i]
            self.population[i].dominated_set_indices = dominated_sets[i]
            if domination_counts[i] == 0:
                self.population[i].rank = 1
                fronts[0].append(self.population[i])
        
        rank = 1
        while len(fronts[rank-1]) > 0:
            next_front = []
            for p in fronts[rank-1]:
                # We iterate based on indices stored
                for q_idx in p.dominated_set_indices:
                    q = self.population[q_idx]
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = rank + 1
                        next_front.append(q)
            rank += 1
            fronts.append(next_front)
            
        if len(fronts[-1]) == 0: del fronts[-1]
        return fronts

    # --------------------------------------------
    # Crowding Distance
    # --------------------------------------------
    def calc_crowding_distance(self, fronts):
        for front in fronts:
            l = len(front)
            if l == 0: continue
            for ind in front: ind.crowding_dist = 0
            if l <= 2:
                for ind in front: ind.crowding_dist = float('inf')
                continue
            
            objs = np.array([ind.objectives for ind in front])
            n_obj = 2
            for m in range(n_obj):
                sorted_idx = np.argsort(objs[:, m])
                front[sorted_idx[0]].crowding_dist = float('inf')
                front[sorted_idx[-1]].crowding_dist = float('inf')
                
                obj_min = objs[sorted_idx[0], m]
                obj_max = objs[sorted_idx[-1], m]
                if obj_max == obj_min: continue
                
                norm = obj_max - obj_min
                prev_vals = objs[sorted_idx[:-2], m]
                next_vals = objs[sorted_idx[2:], m]
                dists = (next_vals - prev_vals) / norm
                
                for i, d in enumerate(dists):
                    idx = sorted_idx[i+1]
                    front[idx].crowding_dist += d

    # --------------------------------------------
    # Binary Tournament
    # --------------------------------------------
    def binary_tournament(self, n_select):
        offspring_pop = MomaPopulation()
        pop_size = len(self.population)
        competitors = np.random.randint(0, pop_size, (n_select, 2))
        
        for i in range(n_select):
            idx_a, idx_b = competitors[i]
            a = self.population[idx_a]
            b = self.population[idx_b]
            
            if a.rank < b.rank: winner = a
            elif b.rank < a.rank: winner = b
            else: winner = a if a.crowding_dist > b.crowding_dist else b
            
            offspring_pop.append(MomaIndividual(winner.assignment.copy()))
        return offspring_pop
    
    # --------------------------------------------
    # Survivor Selection
    # --------------------------------------------
    def survivor_selection(self, pop_size):
        fronts = self.calc_domination_rank()
        self.calc_crowding_distance(fronts)
        new_pop = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                new_pop.extend(front)
            else:
                front.sort(key=attrgetter('crowding_dist'), reverse=True)
                new_pop.extend(front[:(pop_size - len(new_pop))])
                break
        self.population = new_pop