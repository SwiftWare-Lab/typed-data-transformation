#!/usr/bin/env python3

def print_flowchart():
    flowchart = r"""
 ┌───────────────────────────────────────────────────────┐
 │                   START                              │
 │                (Read Dataset D)                      │
 └───────────────────────────────────────────────────────┘
                 |
                 v
 ┌───────────────────────────────────────────────────────┐
 │       Partition Dataset into Fixed-Size Blocks       │
 │           e.g., blocks of 65,536 bytes               │
 └───────────────────────────────────────────────────────┘
                 |
                 v
 ┌───────────────────────────────────────────────────────┐
 │   For Each Byte Position (bᵢ): Form a “Byte Group”   │
 │  (Collect all bytes at position i across all blocks) │
 └───────────────────────────────────────────────────────┘
                 |
                 v
 ┌───────────────────────────────────────────────────────┐
 │                Feature Extraction                    │
 │  - Compute Entropy-based metrics (H̄, σ(H), max(H),   │
 │    min(H)) for each Byte Group                       │
 │  - Compute 256-dim Normalized Frequency Vector fᵢ(j) │
 └───────────────────────────────────────────────────────┘
                 |
                 v
 ┌───────────────────────────────────────────────────────┐
 │   Combine All Features into a Single Feature Vector   │
 │         xᵢ = [H̄ᵢ, σ(Hᵢ), max(Hᵢ), min(Hᵢ), fᵢ(0),        │
 │                  fᵢ(1), …, fᵢ(255)] ∈ ℝ²⁶⁰            │
 └───────────────────────────────────────────────────────┘
                 |
                 v
 ┌───────────────────────────────────────────────────────┐
 │           Hierarchical Clustering (Agglomerative)    │
 │             - Use Complete Linkage Distance          │
 │             - Distance d(x, y) = ‖x - y‖₂             │
 │             - Build Dendrogram & Linkage Matrix       │
 └───────────────────────────────────────────────────────┘
                 |
                 v
 ┌───────────────────────────────────────────────────────┐
 │      Internal Validation & Automatic Cluster Cut      │
 │   - Evaluate Silhouette, Davies-Bouldin, etc.         │
 │   - Determine # of Clusters that Optimizes Metrics    │
 └───────────────────────────────────────────────────────┘
                 |
                 v
 ┌───────────────────────────────────────────────────────┐
 │            Output Optimal Clusters &                 │
 │      Reordered Byte Positions for Compression         │
 └───────────────────────────────────────────────────────┘
                 |
                 v
 ┌───────────────────────────────────────────────────────┐
 │                         END                          │
 └───────────────────────────────────────────────────────┘
"""
    print(flowchart)

if __name__ == "__main__":
    print_flowchart()
