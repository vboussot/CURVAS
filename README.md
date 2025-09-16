[![Grand Challenge](https://img.shields.io/badge/Grand%20Challenge-CURVAS-blue)](https://curvas.grand-challenge.org/) [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-CURVAS-orange)](https://huggingface.co/VBoussot/Curvas)

# CURVAS 2025: Bayesian Uncertainty with Pre-trained TotalSegmentator (3rd place 🥈)

This repository provides the code, configurations, and models used for our submission to the **first edition of the CURVAS Challenge**, focused on **uncertainty-aware segmentation** of **abdominal organs (pancreas, kidneys, liver)** from CT scans.  
Our approach extends the **pre-trained TotalSegmentator model** with **Adaptable Bayesian Neural Network (ABNN)** layers to model **inter-annotator variability** and produce **voxel-wise uncertainty maps**.

---
## 🏁 Official Challenge Results

Our method ranked **3rd overall** in the first edition of the CURVAS Challenge 🏅

| Rank | Team / Algorithm            | DSC (↑)  | thresh-DSC (↑) | ECE (↓)   | CRPS (↓)   |
|------|-----------------------------|----------|----------------|-----------|------------|
| 🏅 1st | SHUGOSHA (MedIG) – curvas_zx | **0.9457** | **0.9787**     | 0.0018    | **8108.48** |
| 🥇 2nd | andreaprenner (PrAEcision!) – nnUNet | 0.9329   | 0.9718         | 0.0022    | 10438.17   |
| 🥈 3rd | **ours (BreizhSeg – TS_BNN)**       | 0.9260   | 0.9717         | **0.0016** | 12325.77   |

✨ Highlights of our approach:
- **Best calibration (ECE)** across all teams  
- Competitive Dice and thresh-DSC, close to top performers  
- Efficient Bayesian extension of pre-trained TotalSegmentator  
