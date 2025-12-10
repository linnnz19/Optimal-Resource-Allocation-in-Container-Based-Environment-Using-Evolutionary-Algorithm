# To run this shell script: bash EC_final_project_Group3.sh

# Change the terminal to Git Bash in VS Code or any IDE you are using.
# If you don't have download Git Bash, download here: https://git-scm.com/downloads
# Or you can simply run the commands one by one in your terminal.

echo "=========================================="
echo "Group 3 - EC Final Project"
echo "Optimal Resource Allocation in Container-Based Environment Using Evolutionary Algorithm"
echo "=========================================="

# Initialize Directories
echo "[Init] Creating output directories..."
mkdir -p Phase1_Output
mkdir -p Phase2_Output
mkdir -p Phase3_Output

# ------------------------------------------
# Phase 1: Cluster Simulation
# ------------------------------------------
echo ""
echo "[Phase 1] Running Cluster Simulation..."
python phase1_cluster_sim.py -i device_spec.cfg

# ------------------------------------------
# Phase 2: Microservice Simulation
# ------------------------------------------
echo ""
echo "[Phase 2] Running Microservice Simulation..."
python phase2_microservice_sim.py -i device_spec.cfg

# ------------------------------------------
# Phase 3: MOMA Optimization
# This configuration results in a total of 100,000 evaluations (20 windows * 50 generations * 100 individuals)
# So it takes a bit longer to run... be patient! (We already use the master-slave structure)
# ------------------------------------------
# Reads Phase 1 & 2 CSVs from the current directory.
echo ""
echo "[Phase 3] Running MOMA Optimization..."
python phase3_moma.py -i device_spec.cfg

# ------------------------------------------
# File Organization (Cleanup)
# ------------------------------------------
echo ""
echo "Moving files to respective folders..."

# Move Phase 1 Output (-f forces overwrite if file exists)
mv -f phase1_cluster_metrics.csv Phase1_Output/
mv -f phase1_network_topology.csv Phase1_Output/
mv -f phase1_heterogeneous_clusters.png Phase1_Output/

# Move Phase 2 Output
mv -f phase2_microservices.csv Phase2_Output/
mv -f phase2_service_interactions.csv Phase2_Output/
mv -f phase2_microservices.png Phase2_Output/
mv -f phase2_interactions.png Phase2_Output/

# Move Phase 3 Output
mv -f phase3_comparison_results.csv Phase3_Output/
mv -f phase3_detailed_allocations.csv Phase3_Output/
mv -f phase3_node_load_details.csv Phase3_Output/
mv -f phase3_comparison_cpu.png Phase3_Output/
mv -f phase3_comparison_mem.png Phase3_Output/
mv -f phase3_comparison_cost.png Phase3_Output/

echo ""
echo "=========================================="
echo "             ALL DONE!"
echo "=========================================="