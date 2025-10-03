# DBLP Link Prediction

Prédiction de futures collaborations entre chercheurs avec GNN + LSTM.

## Quick Start

**Un seul fichier à utiliser:**

```bash
jupyter notebook DBLP_Link_Prediction_Complete.ipynb
```

Exécutez les cellules dans l'ordre. Le notebook contient **TOUT**:
- Extraction des données DBLP → `co_publications.csv`
- Analyse complète répondant aux 7 questions
- Modèle GCN + LSTM
- Métriques et visualisations

## 📋 Structure

**PARTIE 1**: Extraction de données
- Téléchargement DBLP XML
- Parsing et génération CSV
- ⚠️ Si vous avez déjà le CSV, sautez cette partie

**PARTIE 2**: Analyse
1. Revue de littérature (statique vs dynamique)
2. Défis des graphes dynamiques
3. Architecture GCN/GAT + LSTM
4. Feature engineering
5. Validation temporelle
6. Métriques appropriées
7. Visualisations

## Fichiers

- `DBLP_Link_Prediction_Complete.ipynb` - **Notebook principal** (tout-en-un)
- `README.md` - Ce fichier
- `research_answers.md` - Réponses détaillées en français

## Installation

```bash
pip install torch torch-geometric networkx pandas numpy scikit-learn matplotlib seaborn tqdm lxml
```

## Points Clés

- **Temporal modeling > Static**: LSTM améliore GCN statique
- **Validation temporelle obligatoire**: Évite data leakage
- **Métriques adaptées**: ROC-AUC, AP, Precision@K (pas accuracy!)
- **Features riches**: Structure + temps + publications

## Résultats Attendus

- ROC-AUC: ~0.75-0.85
- Average Precision: ~0.70-0.80
- Precision@100: ~0.40-0.60
