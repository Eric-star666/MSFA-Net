Title:MSFA-Net: An Advanced Deep Learning Model for Identifying Blue Horizontal-Branch Stars from LAMOST DR12

Authors: Mingyuan Wang, Xiaoming Kong, Jie Ju, Yude Bu, and Yuchen Liang

Description of contents: The function of each file
·MSFA-Net Model:
·msfa_net.py: Stage-1 pre-training code. It implements the MSFA-Net architecture for 5-class stellar spectral classification and saves the base feature weights.
·msfanet_stage2_finetune.py: Stage-2 fine-tuning code. It loads the pre-trained weights, adapts the model for binary classification (BHB vs. Non-BHB), and applies the Weighted Label-Smoothing Cross-Entropy (W-LSCE) loss.