# DBLP Link Prediction

Predicting future collaborations between researchers with GNN + LSTM.

## Quick Start

**Single file to use:**

```bash
jupyter notebook DBLP_Link_Prediction_Complete.ipynb
```

Run the cells in order. The notebook contains **EVERYTHING**:
- DBLP data extraction → `co_publications.csv`
- Complete analysis answering 7 questions
- GCN + LSTM model
- Metrics and visualizations

## 📋 Structure

**PART 1**: Data extraction
- DBLP XML download
- Parsing and CSV generation
- ⚠️ If you already have the CSV, skip this part

**PART 2**: Analysis
1. Literature review (static vs dynamic)
2. Dynamic graph challenges
3. GCN/GAT + LSTM architecture
4. Feature engineering
5. Temporal validation
6. Appropriate metrics
7. Visualizations

## Files

- `DBLP_Link_Prediction_Complete.ipynb` - **Main notebook** (all-in-one)
- `README.md` - This file
- `research_answers.md` - Detailed answers in French

## Installation

```bash
pip install torch torch-geometric networkx pandas numpy scikit-learn matplotlib seaborn tqdm lxml
```

## Key Points

- **Temporal modeling > Static**: LSTM improves static GCN
- **Temporal validation mandatory**: Avoids data leakage
- **Appropriate metrics**: ROC-AUC, AP, Precision@K (not accuracy!)
- **Rich features**: Structure + time + publications

## Expected Results

- ROC-AUC: ~0.75-0.85
- Average Precision: ~0.70-0.80
- Precision@100: ~0.40-0.60
