# Machine Learning on Player-Generated Game Data: Skill, Strategy and Sketch Recognition

## Overview

This project explores how different machine-learning techniques can learn from different forms of **player-generated game data**.

Rather than using a single type of input, the project studies three complementary datasets representing:

- **Player behaviour** — how a player performs during gameplay
- **Player strategy** — the choices made when forming a team
- **Player-created visual data** — sketches produced during a game

The overall aim is to investigate how the structure and representation of game-generated data influences the suitability and performance of different machine-learning approaches.

## Research Aim

> **How effectively can classical and neural machine-learning methods learn from different representations of player-generated game data?**

The three datasets form separate classification tasks:

1. **SkillCraft1** — predict a player's StarCraft II skill level from gameplay behaviour.
2. **Dota2 Games Results** — predict the winning team from hero selections and match settings.
3. **Google Quick, Draw!** — classify a player's sketch into one of eight object categories.

---

## Datasets

### 1. SkillCraft1 Master Table

**Task:** Multiclass / ordinal player-skill classification

- **Examples:** 3,395 players
- **Target:** `LeagueIndex`
- **Classes:** 8 skill levels, from Bronze to Professional
- **Raw columns:** 20
- **Current clean feature set:** 15 numerical gameplay features
- **Feature examples:** APM, ActionLatency, NumberOfPACs, hotkey usage, minimap activity, map exploration, and unit-production behaviour

Initial EDA identified missing values represented by `?` in:

- `Age` — 55 missing values
- `HoursPerWeek` — 56 missing values
- `TotalHours` — 57 missing values

These variables were excluded from the initial clean modelling feature set because the missingness is heavily concentrated in the Professional class.

Initial analysis also found clear relationships between gameplay behaviour and skill. For example, median APM rises substantially across the league hierarchy, while median ActionLatency decreases.

**Source:**  
UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/272/skillcraft1%2Bmaster%2Btable%2Bdataset

**DOI:**  
https://doi.org/10.24432/C5161N

**Licence:** CC BY 4.0

**Related literature:**  
Thompson, J.J., Blair, M.R., Chen, L. and Henrey, A.J. (2013). *Video Game Telemetry as a Critical Tool in the Study of Complex Skill Learning*. PLOS ONE, 8(9), e75129.  
https://doi.org/10.1371/journal.pone.0075129

---

### 2. Dota2 Games Results

**Task:** Binary match-outcome classification

- **Examples:** 102,944 matches
  - 92,650 training matches
  - 10,294 test matches
- **Target:** `win`
  - `1` = Team 1 wins
  - `-1` = Team 2 wins
- **Columns:** 117 total
- **Hero-selection features:** 113
- **Additional match fields:** `clusterid`, `gamemode`, and `gametype`

Each hero feature is encoded as:

- `1` — hero selected by Team 1
- `-1` — hero selected by Team 2
- `0` — hero not selected

Every match contains exactly **10 selected heroes**, with **5 heroes per team**.

The hero-selection representation is highly sparse because only 10 of the 113 possible hero features are active in any one match. Initial EDA found approximately **91.15% sparsity** in the hero matrix.

The training target is reasonably balanced:

- Team 1 wins: 48,782
- Team 2 wins: 43,868

There are no missing values in the training data.

**Source:**  
UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/367/dota2%2Bgames%2Bresults

**DOI:**  
https://doi.org/10.24432/C5W593

**Licence:** CC BY 4.0

---

### 3. Google Quick, Draw!

**Task:** Multiclass image classification

The original Quick, Draw! dataset contains millions of sketches produced by players of Google's Quick, Draw! game. For this project, a balanced subset was created.

**Selected classes:**

- airplane
- apple
- bicycle
- cat
- fish
- guitar
- house
- tree

**Project subset:**

- **Examples:** 24,000 drawings
- **Classes:** 8
- **Examples per class:** 3,000
- **Image size:** 28 × 28 grayscale
- **Features per image:** 784 pixel values

A fixed random seed (`42`) was used to randomly sample 3,000 drawings from each class for reproducibility.

Preprocessing included:

- balancing the dataset to 3,000 examples per class
- reshaping flat 784-element vectors into `28 × 28 × 1` images
- normalising pixel values from `0–255` to `0–1`
- stratified train/validation/test splitting

Final split:

- **Training:** 19,200 images
- **Validation:** 2,400 images
- **Test:** 2,400 images

Each split preserves equal representation of all eight classes.

**Source:**  
Google Creative Lab — Quick, Draw! Dataset  
https://github.com/googlecreativelab/quickdraw-dataset

**Licence:** CC BY 4.0

**Related literature:**  
Ha, D. and Eck, D. (2017). *A Neural Representation of Sketch Drawings*.  
https://arxiv.org/abs/1704.03477

The dataset was also used in Google's 2018 **Quick, Draw! Doodle Recognition Challenge** on Kaggle.

---

## Project Structure

```text
F21DL-Group-13-Project/
│
├── data/
│   └── README.md
│
├── figures/
│   ├── skillcraft_league_distribution.png
│   ├── skillcraft_apm_latency_by_league.png
│   ├── skillcraft_correlation_heatmap.png
│   ├── dota2_match_distributions.png
│   ├── dota2_top15_hero_usage.png
│   ├── quickdraw_sample_grid.png
│   └── quickdraw_class_distribution.png
│
├── notebooks/
│   ├── 01_SkillCraft_EDA.ipynb
│   ├── 02_Dota2_EDA.ipynb
│   └── 03_QuickDraw_EDA.ipynb
│
├── .gitignore
├── DATA_SOURCES.md
├── Machine Learning for Video Game Classification.pdf
├── README.md
└── requirements.txt
```

Raw datasets are intentionally excluded from Git using `.gitignore`. The datasets should be downloaded from their original sources.

---

## Current Exploratory Analysis

### SkillCraft1

The notebook currently covers:

- dataset loading and structural inspection
- data-type investigation
- missing-value analysis
- league/class distribution
- clean feature-matrix construction
- APM and ActionLatency comparisons across leagues
- descriptive statistics
- feature-to-target correlations
- feature correlation heatmap

### Dota2

The notebook currently covers:

- train/test loading
- column naming and structure validation
- target, game-mode, and game-type distributions
- missing-value checks
- hero encoding validation
- team-size validation
- sparsity calculation
- match outcome visualisation
- hero usage analysis

### Quick, Draw!

The notebook currently covers:

- direct downloading of official bitmap files
- image-format validation
- class sampling
- balanced subset construction
- sample image visualisation
- class-distribution visualisation
- normalisation and reshaping
- stratified train/validation/test splitting
- saving the processed subset

---

## Planned Modelling

The project will compare classical machine-learning methods with neural approaches appropriate to each data representation.

### SkillCraft1

Potential models include:

- Decision Tree
- Random Forest
- k-Nearest Neighbours
- Naïve Bayes / Logistic-style baselines
- Multilayer Perceptron (MLP)

### Dota2

Potential models include:

- Logistic Regression
- Decision Tree
- Random Forest
- Multilayer Perceptron (MLP)

### Quick, Draw!

Potential models include:

- k-Nearest Neighbours or tree-based classical baselines on flattened pixels
- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)

The goal is not to compare raw accuracy directly between unrelated tasks, but to evaluate which model families are suitable for each representation of player-generated data.

---

## Setup

Clone the repository and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Then open the notebooks from the `notebooks/` directory.

---

## Reproducibility

Where random sampling is used, a fixed random seed of `42` is used to make the generated subsets reproducible.

For Quick, Draw!, the raw class files are downloaded directly from Google's official storage and are not committed to the repository.

---

## Data and Licensing

This repository does **not** redistribute the original raw datasets.

Please refer to `DATA_SOURCES.md` and `data/README.md` for dataset links, provenance, and licensing information.

Dataset licences:

- **SkillCraft1:** CC BY 4.0
- **Dota2 Games Results:** CC BY 4.0
- **Google Quick, Draw!:** CC BY 4.0

Users should follow the attribution requirements of the original dataset providers.

---

## Project Status

**Current stage:** Dataset validation, preprocessing, exploratory data analysis, and pitch preparation.

The next stage is to implement and evaluate the planned classical and neural machine-learning models across the three datasets.
