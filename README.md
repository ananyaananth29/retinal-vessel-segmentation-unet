
# Retinal Vessel Segmentation with U-Net

This repository contains code and analysis for segmenting blood vessels in retinal images using U-Net, implemented in PyTorch. The project is based on the DRIVE and STARE datasets and explores how skip connections affect model performance in identifying fine vascular structures.

---

##  Project Overview

This project was completed as part of **Assignment 3** for a deep learning course. The task involved implementing, training, and evaluating two U-Net variants:

- **Model 1: U-Net with Skip Connections**  
- **Model 2: U-Net without Skip Connections**

We analyzed how architectural differences influence the segmentation of retinal blood vessels, especially **thin vessels**, which are crucial in early diagnosis of eye diseases.

---

##  Datasets

- **DRIVE**: Digital Retinal Images for Vessel Extraction  
- **STARE**: Structured Analysis of the Retina  

---

##  Features

- Full **U-Net architecture** with configurable depth and skip connections
- **Data augmentation**: horizontal and vertical flipping
- **Weighted loss function** to handle class imbalance (background vs. vessels)
- Visual output comparison between predictions and ground truth
- Evaluation using **F1-score**, with a target ≥ 0.75 on validation

---

##  Technologies Used

- Python 3.x
- PyTorch ≥ 1.7
- NumPy, scikit-learn, scikit-image
- ImageIO, imagecodecs
- Matplotlib, Seaborn

---

##  Results

| Model Variant         | Validation F1 Score | Observations                               |
|----------------------|---------------------|--------------------------------------------|
| U-Net (with skips)   | > 0.81              | Performs well on both thick and thin vessels |
| U-Net (no skips)     | < 0.75              | Misses thin vessels due to lack of detail retention |

---

##  Sample Outputs

| Ground Truth         | Predicted (U-Net)   |
|----------------------|---------------------|
| ![gt](images/gt1.png)| ![pred](images/pred1.png)|

*(Note: Replace with your actual plots if uploading)*

---

##  Key Insights

- **Skip connections are critical** for preserving spatial resolution in segmentation tasks.
- **Weighted loss functions** significantly improve performance on imbalanced datasets.
- **Simple augmentations** like flipping are highly effective for medical image training.

---

##  File Structure

```bash
├── model_ex13.py              # U-Net with skip connections
├── model_ex15.py              # U-Net without skip connections
├── train.py                   # Training loop and evaluation
├── utils.py                   # Data loading, augmentation, etc.
├── Assignment 3 Report.pdf    # Full report with results and analysis
├── README.md                  # This file
````

---

##  Setup Instructions

1. Create a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```


3. Run training:

   ```bash
   sbatch run.slurm
   ```

---

##  Acknowledgments

* Based on coursework at the University of Utah.
* Thanks to the creators of the DRIVE and STARE datasets.
* U-Net architecture by Ronneberger et al. (2015).

---

##  Contact

For any questions or collaboration ideas, feel free to reach out!

**\Ananya Ananth**
\ananya.ananth.jain@gmail.com
\https://www.linkedin.com/in/ananyaananth/
