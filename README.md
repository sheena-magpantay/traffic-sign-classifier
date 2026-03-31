Utilizing Convolutional Neural Network and Computer Vision for Traffic Sign
Recognition
------------------------------------------------------------------------------

| Team Members | Roles | 
|---|---|
| Bacani, Cyril Cris T.| Data & Ethics Lead |
| Egaran, Chelsea Felize B.| Modeling Lead | 
| Magpantay, Sheena Marie B.| Project Lead / Integration | 
|  Manalo, Kisha Margarette B.| Evaluation & MLOps Lead | 

OVERVIEW:
------------------------------------------------------------------------------
This project focuses on utilizing Convolutional Neural Network (CNN), Natural Language Processing (NLP), and Reinforcement Learning (RL)  to
recognize traffic signs in a digital system. The goal of the project is to develop a model that can
detect and classify different traffic signs from images input.The project involves collecting a
dataset of traffic sign images, preprocessing the data, training the CNN model, and evaluating
its performance in recognizing various signs.

## Quickstart

Follow these steps to get the Traffic Sign Recognition system up and running.

### 1. Install Dependencies
### 2. Prepare the Dataset
### 3. Train the CNN Model

```bash
python models/finetune.py
```
Training logs and plots will be saved to:
```
experiments/
├── logs/
└── results/
```

### 4. Run Traffic Sign Recognition

```bash
python models/entrypoint.py
```

---

## Results

After training and evaluation, the following output files will be generated in `experiments/results/`:

| Output | File |
|--------|------|
| Training history | `Training History Plot.png` |
| Confusion matrix | `Confusion Matrix.png` |
| Calibration curve | `Calibration Curve.png` |


