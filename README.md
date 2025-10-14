# DBLP Link Prediction: Predicting Research Collaborations

Predict future collaborations between researchers using Graph Neural Networks (GNN) combined with temporal modeling on the [DBLP co-authorship dataset](https://snap.stanford.edu/data/com-DBLP.html).

<img src="https://github.com/martinoyovo/dblp_link_prediction/blob/main/metrics_curves.png" width="100%">

## Overview

This project implements a complete pipeline for temporal link prediction in co-authorship networks. Starting from raw DBLP XML data, it extracts co-publication relationships, analyzes temporal graph evolution, and builds a GCN-based model to predict future collaborations.

## Quick Start

**Single notebook with everything:**

```bash
jupyter notebook DBLP_Link_Prediction_Complete.ipynb
```

Run cells sequentially. The notebook is organized into two main parts:

### Part 1: Data Extraction
- Downloads DBLP XML (~950MB, takes 10-15 minutes)
- Parses publications and extracts co-authorship links
- Generates `co_publications.csv` with (year, author1, author2) format
- **Skip this if you already have the CSV file**

### Part 2: Analysis & Modeling
Addresses 7 key research questions:

1. **Literature Review**: Why dynamic methods outperform static approaches
2. **Identified Challenges**: Temporality, distribution shift, sparsity, cold start
3. **Architecture Justification**: GCN for structure + LSTM for temporal dynamics
4. **Feature Engineering**: Structural (degree, clustering, PageRank) + temporal features
5. **Temporal Validation**: Why random splits cause data leakage
6. **Appropriate Metrics**: ROC-AUC, Average Precision, Precision@K, Hits@K
7. **Visualizations**: Graph evolution over time + model performance curves

## Dataset Statistics

After extraction, you'll work with:
- **253,921 co-publication records**
- **82,882 unique authors**
- **Time span**: 1951-2026
- **Analysis period**: 2015-2020 (demonstrates temporal evolution)

Example temporal growth:
- 2015: 4,352 edges
- 2020: 53,013 edges

## Installation

```bash
pip install torch torch-geometric networkx pandas numpy scikit-learn matplotlib seaborn tqdm lxml
```

**Requirements:**
- Python 3.9+
- ~2GB disk space for DBLP XML
- Basic familiarity with PyTorch and graph neural networks

## Methodology Highlights

### Why This Approach Works

**Dynamic > Static**: The notebook demonstrates that collaboration networks constantly evolve. Static methods (GCN, GraphSAGE) miss:
- Recurring collaboration patterns
- Publication velocity trends
- Temporal dependencies spanning 20+ years

**GCN + LSTM Architecture**:
```
Temporal Graphs [G₂₀₁₅, G₂₀₁₆, G₂₀₁₇]
    ↓ GCN (captures network structure)
Node Embeddings [h₂₀₁₅, h₂₀₁₆, h₂₀₁₇]
    ↓ LSTM (captures temporal evolution)
Final State h₂₀₁₈
    ↓ MLP
P(link | author_i, author_j)
```

### Critical Implementation Details

**Temporal Validation** (not random splits!):
```
Train: 2015-2017 graphs → Build model
Test:  2019 new edges  → Predict collaborations
```
This prevents data leakage and mirrors real-world deployment.

**Feature Engineering**:
- Structural: Node degree, clustering coefficient, PageRank
- Network: Number of neighbors, collaboration patterns
- All features normalized for stable training

**Metrics for Imbalanced Data**:
Since only ~1% of potential links are positive:
- **ROC-AUC**: Overall ranking quality (0.75-0.85 expected)
- **Average Precision**: Focus on positive class (0.70-0.80 expected)
- **Precision@K**: Accuracy in top-K predictions (0.40-0.60 for K=100)
- **Hits@K**: Coverage of true links in top-K

## Expected Results

<img src="https://github.com/martinoyovo/dblp_link_prediction/blob/main/graph_evolution.png" width="100%">

The notebook generates:
- **Temporal evolution plots**: Nodes, edges, and density over time
- **Performance curves**: ROC and Precision-Recall curves
- **Evaluation metrics**: Complete breakdown of model performance

Typical performance on test set:
- ROC-AUC: 0.75-0.85
- Average Precision: 0.70-0.80
- Precision@100: 0.40-0.60 (40-60 true future collaborations in top 100 predictions)

## File Structure

```
.
├── DBLP_Link_Prediction_Complete.ipynb  # Main notebook (all-in-one)
├── README.md                             # This file
├── research_answers.md                   # Detailed answers (French)
├── co_publications.csv                   # Generated data (after Part 1)
├── metrics_curves.png                    # Performance visualization
└── graph_evolution.png                   # Network dynamics over time
```

## Key Takeaways

- **Temporal modeling is essential** for evolving networks like co-authorship graphs
- **Temporal validation prevents leakage** - never use random train/test splits for temporal data
- **Standard accuracy is misleading** with imbalanced data - use ROC-AUC and Precision@K
- **Feature engineering matters** - combining structural and temporal features improves predictions

## Use Cases

- **Academic networking**: Recommend potential collaborators to researchers
- **Conference planning**: Identify emerging research communities  
- **Funding agencies**: Predict successful research partnerships
- **Trend analysis**: Track evolution of scientific collaboration patterns

## DBLP Dataset Details

From [SNAP](https://snap.stanford.edu/data/com-DBLP.html):
- Comprehensive computer science bibliography
- Co-authorship network: authors connected if they co-published
- Ground-truth communities based on publication venues
- Network contains largest connected component

## Citation

If you use this code, please cite the DBLP dataset:

```bibtex
@inproceedings{snapnets,
  author = {Leskovec, Jure and Krevl, Andrej},
  title = {{SNAP Datasets}: {Stanford} Large Network Dataset Collection},
  howpublished = {\url{http://snap.stanford.edu/data}},
  month = jun,
  year = 2014
}
```

## Notes

- The notebook includes working code examples, not just theoretical discussion
- Data extraction can be slow - be patient during Part 1
- Example visualizations use simulated data for demonstration
- All 7 research questions are addressed with both theory and implementation

---

**Questions?** The notebook contains detailed explanations for each step. For methodology details, see `research_answers.md`.
