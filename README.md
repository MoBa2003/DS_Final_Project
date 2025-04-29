# Search Engine and Document Clustering

This project is a two-phase implementation of a simple **search engine** and **document clustering system** based on **TF-IDF**, **cosine similarity**, and **PCA+KMeans**.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Phase 1: Search Engine](#phase-1-search-engine)
  - [Phase 2: Document Clustering](#phase-2-document-clustering)
- [Output](#output)
- [Notes](#notes)

## Overview
This project processes a corpus of documents and enables two main tasks:
1. **Search Engine:** Given a textual query and a set of candidate document IDs, it computes TF-IDF vectors for both the query and documents, ranks them based on cosine similarity, and returns top matches with key paragraph highlights.
2. **Clustering:** Performs dimensionality reduction (PCA) and unsupervised clustering (KMeans) on the document TF-IDF vectors, visualizing them in a 2D plot.

## Features
- Tokenization, punctuation removal, and lemmatization
- Stopword removal
- TF, IDF, and TF-IDF computation
- Cosine similarity scoring between query and documents
- Ranking and explanation with most frequent and important words
- Paragraph-level relevance matching
- PCA-based dimensionality reduction
- KMeans clustering and visualization

## Project Structure
```
project/
│
├── data/                          # Folder containing documents
│   └── document_{id}.txt          # Individual text files
├── data.json                      # Query and candidate documents information
├── Search_Engine.py              # Phase 1: Search engine implementation
├── phase_2.py                    # Phase 2: Clustering implementation
├── graph.png                     # Output clustering plot
└── README.md                     # Project description
```

## Requirements
Install required Python libraries:
```bash
pip install nltk numpy matplotlib seaborn scikit-learn
```
You must also download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

## Usage
### Phase 1: Search Engine
```bash
python Search_Engine.py
```
- You will be prompted to enter a query.
- It uses `data.json` to find the corresponding query info and candidate document IDs.
- For the top 10 relevant documents, it prints:
  - Repetitive (most frequent) words
  - Important (high TF-IDF) words
  - Top 3 most similar paragraphs

### Phase 2: Document Clustering
```bash
python phase_2.py
```
- Randomly selects 100 documents from `data/`
- Computes reduced TF-IDF vectors (2D) using PCA
- Applies KMeans clustering (5 clusters)
- Saves scatter plot as `graph.png`

## Output
- Ranked document results with matching paragraphs printed in console (Phase 1)
- `graph.png` visualizing document clusters (Phase 2)

## Notes
- Be sure to have at least some document files in the `data/` directory for testing.
- The `data.json` file should follow a structure like:
```json
[
  {
    "query": "sample query",
    "candidate_documents_id": [1, 2, 3, ...]
  },
  ...
]
```
- Closest matches in vocabulary are handled using `difflib.get_close_matches`.

---
Developed as a basic NLP/IR project with visualization support.

