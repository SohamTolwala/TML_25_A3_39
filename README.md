# TML_25_A3_39

# TML Assignment 3 – Model Robustness  

**Team 39** • Token: 09596680  
Soham Ritesh Tolwala – soto00002@stud.uni-saarland.de  
Laxmiraman Dnyaneshwar Gudewar – lagu00003@stud.uni-saarland.de  

---

## Objective

Design a robust classifier on CIFAR-10-like data using adversarial training, particularly PGD, while maintaining competitive performance on:
- Clean data
- FGSM adversarial examples
- PGD adversarial examples

---

## Repository Structure

| File / Folder              | Purpose                                                                 |
|---------------------------|-------------------------------------------------------------------------|
| `colab_training.ipynb`    | Main notebook for adversarial training, experimentation, and tuning. |
| `FREE.ipynb`              | Attempted FREE adversarial training approach (discarded).            |
| `PGD.ipynb`               | Initial PGD attack and defense experiments.                           |
| `try_script.py`           | Raw model submission script to the leaderboard (without assertions). |
| `example_assignment_3.py` | Baseline model submission script (with assertions).                   |
| `resnet_wrapper.py`       | Custom wrapper for defining and exporting ResNet architectures.       |
| `sanity_check.py`         | Validates model accuracy on [0,1] vs [0,255] image inputs.            |
| `./data/Train.pt`         | Provided dataset (wrapped via `TaskDataset` and `DataWrapper`).       |
| `saved_models/`           | Contains trained model checkpoints for final and intermediate runs.  |

---

## Evaluation Results

| Metric             | Final Model Score |
|--------------------|-------------------|
| Clean Accuracy   | 52.43%            |
| FGSM Accuracy    | 38.27%            |
| PGD Accuracy     | 35.83%            |

Model passed server-side checks after several debugging iterations involving input normalization, data composition tuning, and stable PGD ramp-up.

---

## Key Techniques Used

- **PGD Adversarial Training** (gradual ramp-up over epochs)
- **70:30 clean-to-adversarial** data mix
- **Adam Optimizer** with **MultiStepLR scheduler**
- **Normalization** using `transforms.functional.to_tensor()` to ensure [0,1] pixel scale
- **Sanity checks** on clean accuracy using `sanity_check.py` to catch evaluation errors before submission

---

> All training was performed on Google Colab (with GPU), final model checkpoint submitted from local system.
