# **Trajectory Prediction with GRU Networks**

---

## **Overview**

This project aims to **predict the future motion (trajectory) of humans** using sequential position data.

Given a small piece of a person’s recent path, the model attempts to draw out where they’re likely to step next — not just the next moment, but a whole  **future segment** .

The project is built around the  **Stanford Drone Dataset (SDD)** , a large-scale collection of people, cyclists, and vehicles recorded from above. Each trajectory is a sequence of 2D coordinates over time. The task is to learn motion patterns from this data and generate predictions for the next series of steps.

---

## **Project Goals**

* Load and process trajectory sequences from SDD
* Use the past *N* positions as model input
* Predict the next *M* future positions (multi-step prediction)
* Evaluate and visualize predicted vs. true trajectories
* Keep the pipeline simple, lightweight, and runnable on CPU

The immediate aim is to build a reliable baseline before moving on to more advanced models or features.

---

## **Model Architecture (Current Choice)**

The main model used in this stage is a  **GRU-based sequence predictor** 

### **Why GRU?**

* Faster than LSTM, especially on CPU
* Fewer parameters → easier to train
* Surprisingly strong performance for smooth motion data
* Simple enough to understand and experiment with

This makes GRU the best choice for early development, especially without dedicated GPU power.

### **Intended GRU Setup**

* **Input:** sequences of past positions, shape `(seq_len_past, 2)`
* **Hidden size:** 64 units
* **Layers:** 1 or 2 stacked GRUs
* **Output:** future trajectory `(seq_len_future, 2)`
* **Prediction mode:** direct multi-step (one-shot output)

The model will take, for example, **8 past points** and predict the next **12** future points.

---

## **Dataset**

The project uses:

### **Stanford Drone Dataset (SDD)**

* Overhead videos of pedestrians/bikers/carts
* Clean 2D coordinates already provided
* Complex real-world motion patterns
* Ideal for multi-step trajectory prediction

Only the position sequences are used — the project does not process video directly.

Do make sure you download it from [keggle ](https://www.kaggle.com/datasets/aryashah2k/stanford-drone-dataset), the original one from Stanford has 70 fucking gigabytes. This one is supposed to be compressed, no significant data loss, at 1.5 GB.

---

## **Repository Structure (Planned)**

```bash
trajectory-prediction/
│
├── data/
│   ├── raw/    
│   └── processed/  
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── configs/
│   └── default.yaml
│
├── scripts/
│   ├── prepare_data.py
│   └── train_local.sh
│
└── README.md

```

---

## **Expected Outcome**

By the end of development, this project should deliver:

A clean training pipeline

A strong GRU baseline on SDD

Multi-step trajectory predictions

Visual plots showing predicted vs real paths

Comparative evaluations between different architectures.

Vizualizing the paths on the videos. First, we'll only draw on images, since for videos we have to interpolate (we have a 30fps video but not so many annotations, so we have to interpolate which... complicates a bit things. Make sure it works first)

---

## **Setting up**

For this it is highly recommended you use a virtual environment. 

1. Create the virtual environment

Run the following in the project root:
```bash
python3 -m venv .venv
```

2. Activate the environment

Linux / macOS:
```bash
source .venv/bin/activate
```

Windows (PowerShell):
```bash
.\.venv\Scripts\activate
```

After activation, your terminal prompt should include (.venv).

3. Install project dependencies

If a requirements.txt file exists, install everything with:
```bash
pip install -r requirements.txt
```

Otherwise, install packages normally:
```bash
pip install <package-name>
```

Save any new installs with:
```bash
pip freeze > requirements.txt
```
4. Deactivate when finished
```bash
deactivate
```

This returns your shell to the system Python environment.

