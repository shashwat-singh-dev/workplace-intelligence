# 🏢 Workplace Intelligence & Personalization Platform

> An AI/ML-based personalized meeting-room allocation and ranking system that combines deterministic constraint satisfaction with machine learning to recommend the most suitable room for each employee.

---

## 🚀 Problem Statement

Modern workplaces operate with a finite set of meeting rooms shared across hundreds of employees. Employees have distinct preferences — for room type, floor, capacity, start time, and duration — and availability alone does not guarantee a suitable allocation.

High-capacity rooms are scarce. Peak hours create congestion. A system that simply finds *any* available room ignores whether that room is actually appropriate for the employee requesting it.

The core problem this project solves is not just feasibility — it is preference relevance:

> **"Among the rooms that are already feasible, which candidate is the most suitable for this specific employee?"**

This requires separating two concerns:
- **Operational feasibility** — does a room satisfy the hard constraints?
- **Personalization** — among all feasible options, which one best fits this employee's behavior and preferences?

---

## 📌 Project Overview

This platform is built as a **two-phase hybrid system**:

### Phase 1 — Deterministic Allocation Engine
Handles all hard operational constraints:
- Room availability
- Capacity requirements
- Room type preferences
- Floor preferences
- Time flexibility

### Phase 2 — ML Personalization & Ranking Layer
Handles soft, preference-based decisions:
- Employee behavior modeling
- Candidate preference relevance scoring
- Learning-to-rank across feasible candidates
- Workplace resource intelligence

The guiding design principle:

> **"First determine what is feasible. Then determine what is preferable."**

### System Architecture

```
Booking Request
       │
       ▼
┌─────────────────────────────┐
│  Phase 1: Deterministic     │
│  Allocation Engine          │
│  - Availability check       │
│  - Capacity constraint      │
│  - Room type hierarchy      │
│  - Time relaxation          │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       │                │
   0 Candidates    1 Candidate     2+ Candidates
       │                │                │
    SKIPPED        DIRECT           Phase 2
                  ALLOCATION      ML Ranking
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  CatBoostRanker  │
                              │  Best Candidate  │
                              └─────────────────┘
                                       │
                                       ▼
                              Room Recommendation
```

---

## 🏢 Simulated Workplace & Dataset

The project operates on a fully simulated workplace environment built to reflect realistic enterprise booking behavior.

| Component | Value |
|---|---:|
| Employees | 1,000 |
| Meeting Rooms | 50 |
| Floors | 5 |
| Total Booking Requests | 40,000 |
| Business Dates | 2025 |

**Room Types Available:**

| Room Type | Description |
|---|---|
| Meeting | Standard meeting rooms |
| Conference | Larger conference spaces |
| Training | Training and workshop rooms |
| Focus | Quiet, focused work rooms |

> ⚠️ Room capacity is treated as a **hard constraint** throughout the system. A room is never allocated if its capacity is insufficient for the requested headcount.

---

## ⚙️ Deterministic Allocation Logic

The allocation engine applies a strict five-level room preference hierarchy before relaxing any time constraints.

### Room Preference Hierarchy

| Level | Logic |
|---|---|
| 1 | Exact room type + exact requested floor |
| 2 | Exact room type + nearby floor |
| 3 | Exact room type + any floor |
| 4 | Similar room type + exact requested floor |
| 5 | Similar room type + any floor |

### Time Relaxation

Time relaxation is applied **only after all five room hierarchy levels have been exhausted** at the original time slot.

- Relaxation increment: **15 minutes**
- Direction: within employee's stated flexibility window
- Capacity constraint: **remains hard throughout**

This ensures that room type and floor preferences are always prioritized over time flexibility.

---

## 📊 Phase 1 Results

| Metric | Value |
|---|---:|
| Total Requests | 40,000 |
| Successfully Allocated | 32,442 |
| Skipped | 7,558 |
| Allocation Success Rate | **81.11%** |

**Why were some requests skipped?**

The majority of skipped requests (75.1%) were caused by scarcity of high-capacity rooms. Rooms with capacity 10, 12, and 16 were in limited supply, and peak-hour demand created unavoidable conflicts that no amount of time relaxation could resolve within each employee's flexibility window.

> The 81.11% booking success rate reflects **real operational constraints**, not a system limitation.

---

## 🧠 ML Personalization

The deterministic allocator is highly effective at resolving operational constraints. However, when **two or more feasible candidates exist**, the allocator selects the first match in its hierarchy — without considering which candidate is most suitable for the individual employee.

This is the gap that machine learning fills.

### Candidate Decision Logic

```
Feasible Candidates Found
         │
    ┌────┴────┐
    │         │
  = 0       = 1          ≥ 2
    │         │            │
 SKIPPED  DIRECT     ML RANKING
          ALLOC      (CatBoostRanker)
```

The ML model ranks all feasible candidates and selects the one with the highest predicted preference relevance for that employee — going beyond what the rule-based system alone can determine.

---

## 🔎 Candidate Generation

The historical bookings table stored only the **final selected room** for each booking — not the other rooms that were available at the time. This meant the feasible candidate space was not directly available for ML training.

To solve this, the project **reconstructed the candidate space** by replaying the deterministic allocation logic chronologically over all historical bookings.

Each row in the candidate dataset represents:

```
1 Employee + 1 Booking Request + 1 Feasible Candidate Room/Time
```

| Metric | Value |
|---|---:|
| Total Candidate Rows | 61,784 |
| Booking Requests with Candidates | 32,442 |
| Employees Covered | 1,000 |
| Candidates per Booking | 1–22 |

> Candidate generation is **not** a second allocator. It reconstructs the feasible option space using the same operational constraints as the allocator, enabling supervised ranking model training.

---

## 👤 Employee Behavior Modeling

A key insight of this project is that employees exhibit **consistent behavioral patterns** over time — preferred room types, preferred floors, preferred hours, and preferred capacity ranges. These patterns are the primary signal for personalization.

### Long-Term Behavior Features
- Room-type preference distribution
- Floor preference distribution
- Capacity behavior (typical vs. requested)
- Start-hour preference distribution
- Behavioral consistency scores

### Recent Behavior Features
- Recent room-type tendencies (behavioral shifts)
- Recent floor preferences
- Recent start-hour patterns

### Employee–Candidate Relationship Features
- Historical booking count with this specific room
- Historical usage percentage for this room type
- Historical usage percentage for this floor
- Historical usage percentage for this start hour
- Previous direct usage of this candidate room

These features capture not just general preference but the **specific historical relationship** between an employee and a particular candidate room.

---

## 🧩 Feature Engineering

Features are organized into four categories:

### Employee Behavior
Long-term and recent behavioral signals derived from each employee's booking history — room type, floor, capacity, and time-of-day preferences.

### Employee–Candidate Relationship
Historical interaction signals between the employee and each specific candidate room — usage frequency, percentage affinity, and prior direct usage.

### Request–Candidate Relationship
Signals comparing the current booking request to the candidate room:

| Feature | Description |
|---|---|
| `capacity_difference` | Difference between candidate capacity and requested headcount |
| `time_difference_minutes` | Time offset between candidate slot and originally requested time |

> `floor_distance` and `room_type_match` were intentionally **excluded** — these signals are already encoded structurally through the deterministic allocation hierarchy and would introduce redundancy.

### Resource Intelligence
Workplace-level scarcity and demand signals:

| Feature | Description |
|---|---|
| `room_historical_demand` | Overall historical demand for this room |
| `room_time_demand` | Demand for this room at this specific hour |
| `room_utilization_rate` | How heavily this room is utilized overall |
| `capacity_feasible_alternative_count` | Number of rooms in the building that satisfy the capacity requirement |

---

## 🔐 Data Leakage Prevention

Leakage prevention is one of the most critical and carefully implemented aspects of this project.

All historical features — employee behavior and employee–candidate relationships — are computed **strictly from past allocations only**. No future information is allowed to influence earlier requests.

### Chronological Feature Pipeline

```
Previous Allocations
        │
        ▼
Generate Historical Features
        │
        ▼
Current Booking Request
        │
        ▼
Generate Candidate Features
        │
        ▼
Model Ranking Decision
        │
        ▼
Update Historical State
        │
        ▼
Next Request
```

### Train / Validation / Test Split

- Split strategy: **Chronological, booking-level**
- All candidate rows belonging to the same booking are **kept together** — they are never split across train/test boundaries
- A random row-level split would have caused leakage by allowing future booking context to appear in training data

---

## 🎯 Ranking Target

One of the most important design decisions in this project was **not** training the model to imitate what the historical allocator had selected.

Had the allocator's selection been used as the training label, the model would simply learn to replicate the deterministic rule system — not to personalize.

Instead, the ranking target is a **continuous preference-relevance score** built from the employee's underlying preferences, computed independently of what the allocator actually chose.

### Relevance Score Components

| Dimension | Signal |
|---|---|
| Room-type alignment | Does this room type match the employee's preferred type? |
| Floor preference | Does this floor match the employee's floor preference? |
| Time preference | Does this time slot align with the employee's typical booking hour? |
| Capacity preference | Does this capacity match the employee's typical headcount? |

This target measures **how well a candidate fits the employee** — not whether the allocator happened to pick it.

---

## 🤖 Machine Learning Model

The project uses **CatBoostRanker** with the **YetiRank** loss function — a learning-to-rank formulation appropriate for ordering multiple candidates per booking request.

### Why Ranking Instead of Classification?

Each booking has multiple feasible candidates that must be ordered by preference relevance. This is inherently a ranking problem, not a binary classification problem. A classifier would treat candidates independently; a ranker models their relative ordering within each booking group.

### Model Configuration

| Parameter | Value |
|---|---:|
| Loss Function | YetiRank |
| Iterations | 500 |
| Learning Rate | 0.05 |
| Depth | 8 |
| Random Seed | 42 |
| L2 Regularization | 5 |
| Evaluation Metric | NDCG@3 |
| Final Trees Used | 488 |

---

## 🏆 Model Performance

Evaluated on a held-out chronological test set:

| Metric | Value |
|---|---:|
| Test Booking Groups | 4,883 |
| **Top-1 Ranking Accuracy** | **90.17%** |
| **Top-3 Hit Rate** | **98.55%** |
| **Mean NDCG@3** | **0.9770** |

> ⚠️ **Important clarifications:**
> - **Top-1 Accuracy** measures whether the model's top-ranked candidate matches the highest-relevance candidate — it is **not** the booking success rate.
> - **CatBoost ranking scores can be negative** — they are relative ordering scores, not probabilities.
> - The model evaluates **which candidate to recommend among feasible options**, not whether a room exists.

---

## 🔬 Personalization vs Resource Intelligence

An ablation study compared a personalization-only model against the full model that includes resource intelligence features.

| Metric | Personalization Only | Full Model |
|---:|---:|---:|
| Top-1 Accuracy | 90.01% | **90.17%** |
| Top-3 Hit Rate | 98.44% | **98.55%** |
| NDCG@3 | 0.9766 | **0.9770** |
| Mean Target Relevance | 0.6767 | **0.6773** |

**Conclusion:** Employee behavior and personalization features provide the dominant ranking signal. Resource-intelligence features (scarcity, demand, utilization) provide a smaller but consistent additional contribution across all metrics.

---

## ⚖️ ML vs Historical Allocator

Comparing the ML model's selections against what the historical deterministic allocator had chosen:

| Metric | Value |
|---|---:|
| Historical Allocator Mean Relevance | 0.6453 |
| ML Model Mean Relevance | 0.6773 |
| Mean Improvement | **+0.0320** |

| Comparison Outcome | % of Bookings |
|---|---:|
| ML selected higher-relevance candidate | 16.67% |
| Same relevance selected | 78.99% |
| Historical allocator selected higher | 4.34% |

The model improves assignment quality in roughly 1 in 6 bookings while matching the allocator in the vast majority — confirming it is not simply imitating the rule system.

---

## 📌 Business Impact

| Metric | Phase 1 Only | Full System (Phase 1 + ML) |
|---|---:|---:|
| Booking Success Rate | 81.11% | 81.11% |
| Mean Preference Relevance | 0.6453 | 0.6773 |
| Relevance Improvement | — | **+4.96% relative** |

**Why does the booking success rate remain the same?**

The ML layer ranks candidates that are **already feasible**. It does not create new availability. Bookings that fail do so because no feasible candidate exists — the ML model has no candidates to rank in those cases.

The system's two roles are clearly separated:

| Phase | Responsibility |
|---|---|
| Phase 1 | Feasibility, constraint satisfaction, booking success |
| Phase 2 | Personalization, assignment quality, preference alignment |

> The ML layer improves **who gets which room** — not whether a room exists at all.

---

## 💡 Key Insights

- **Hybrid systems work well** — separating hard constraints from soft preferences allows each layer to do what it does best
- **Candidate generation is non-trivial** — feasible alternatives must be reconstructed when only final outcomes are logged
- **Chronological feature engineering is essential** — generating historical features without time-awareness introduces silent data leakage
- **Learning-to-rank outperforms classification** for multi-candidate preference problems
- **Personalization is the dominant signal** — employee behavior features contribute far more than scarcity alone
- **Resource intelligence adds context** — workplace-level signals provide a small but consistent improvement
- **Assignment quality and booking volume are separate problems** — improving one does not automatically improve the other

---

## ⚡ Challenges & Solutions

| Challenge | Solution Implemented |
|---|---|
| Feasible candidates not stored in historical data | Chronological candidate reconstruction using the same allocator logic |
| Data leakage in historical features | Strict chronological feature generation — future allocations cannot influence earlier requests |
| Model imitating allocator rather than personalizing | Non-imitative preference-relevance ranking target, independent of allocator selections |
| Cold-start employees with no history | Behavioral fallback defaults for employees with insufficient booking history |
| Slow availability checking at scale | Binary search + `room_last_end` fast-path optimization (27.7× speedup) |
| Balancing personalization with resource signals | Ablation study to isolate contribution of each feature group |
| Deploying without re-scanning historical data | Precomputed behavior and scarcity lookup tables for inference |

---

## 📚 Key Learnings

`Machine Learning` · `Learning-to-Rank` · `Personalization` · `Recommendation Systems` · `Feature Engineering` · `Employee Behavior Modeling` · `Data Leakage Prevention` · `Chronological Validation` · `NDCG` · `Ablation Studies` · `Resource Intelligence` · `CatBoost` · `ML Inference Pipelines` · `Streamlit Deployment` · `Git / GitHub`

---

## 🖥️ Application Features

The deployed Streamlit application provides an interactive interface for the full two-phase pipeline.

### Input Fields

| Field | Description |
|---|---|
| Employee ID | Identifies the employee and loads their behavior profile |
| Meeting Date | Target date for the booking |
| Start Time | Requested start time |
| Duration | Required meeting duration |
| Room Type | Preferred room type |
| Preferred Floor | Preferred floor |
| Required Capacity | Minimum headcount requirement |
| Time Flexibility | Maximum allowable time relaxation |

### Application Flow

1. Validate employee ID and load behavior profile
2. Generate all feasible candidates using deterministic logic
3. Apply hard capacity and availability constraints
4. If one candidate → direct allocation
5. If multiple candidates → ML ranking via CatBoostRanker
6. Display the recommended room
7. Display allocation details, ranking scores, and candidate breakdown

---

## 📸 Application Screenshot

![Workplace Intelligence Application](assets/project_screenshot.png)

---

## 🚀 Deployment

The application uses precomputed lookup tables during inference to avoid repeatedly scanning the full historical dataset at runtime.

### Inference Architecture

```
User Input
    │
    ▼
Streamlit Interface
    │
    ▼
Candidate Generation (Deterministic Allocator)
    │
    ▼
Employee Behavior Lookup
    │
    ▼
Resource Intelligence Lookup
    │
    ▼
CatBoostRanker (room_ranking_model.cbm)
    │
    ▼
Best Candidate → Room Recommendation
```

### Live Application

🔗 [workplace-intelligence-shashwat.streamlit.app](https://workplace-intelligence-shashwat.streamlit.app)

---

## 🛠️ Tech Stack

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- CatBoost (CatBoostRanker)
- Scikit-learn
- Learning-to-Rank
- Feature Engineering
- Model Evaluation (NDCG@3, Top-K Accuracy)

### Application & Deployment
- Streamlit
- Streamlit Cloud

### Tools
- Jupyter Notebook
- VS Code
- Git
- GitHub

---

## 📂 Project Structure

```
Workplace Intelligence/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   ├── employees.csv
│   │   ├── rooms.csv
│   │   ├── room_booking_requests.csv
│   │   └── room_bookings.csv
│   │
│   └── processed/
│       ├── room_type_behavior_features.csv
│       ├── floor_behavior_features.csv
│       ├── capacity_behavior_features.csv
│       └── start_hour_behavior_features.csv
│
├── models/
│   ├── room_ranking_model.cbm
│   ├── feature_schema.json
│   ├── employee_behavior_lookup.csv
│   ├── room_scarcity_lookup.csv
│   └── room_time_scarcity_lookup.csv
│
├── notebooks/
│   ├── 01_Long_term_behaviour.ipynb
│   ├── 02_Recent_behaviour.ipynb
│   ├── 03_ML_model.ipynb
│   └── 04_ML_Ranking.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/workplace-intelligence.git
cd workplace-intelligence
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application
```bash
streamlit run app/app.py
```

The application will open in your browser at `http://localhost:8501`.

> Alternatively, use the live deployed version:
> 🔗 [workplace-intelligence-shashwat.streamlit.app](https://workplace-intelligence-shashwat.streamlit.app)

---

## 📄 License

This project is for educational and portfolio purposes.

---

*Built by Shashwat Singh · B.Tech CSE (AI & ML) · LNCT Bhopal*
