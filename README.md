# ⚠️ Important Note on SATA Evaluation Results
> **This CVPR-level publication will be more solid if authors could address several critical issues in the evaluation.**

https://github.com/nick-nikzad/SATA/issues/2#issuecomment-3112310860

The method achieves 95% top-1 accuracy on the ImageNet-1K validation set, but drops to 50% after shuffling. This suggests SATA may be exploiting the default ordering of the dataset—where samples are sorted by class (e.g., the first 50 images are class 0, the next 50 are class 1, and so on)—rather than truly learning from image content.
<!-- ![Image](https://github.com/user-attachments/assets/0a755615-fccb-4944-b662-683b25a05fdf) -->

![Image](img/image1.png)

Shuffling is a way to break this default ordering, and the performance drops a lot.

<!-- ![Image](https://github.com/user-attachments/assets/f4c9d2a6-b602-456e-878d-797f175ae3e1) -->

![Image](img/image2.png)

https://github.com/nick-nikzad/SATA/issues/2#issuecomment-3112310860

I did the experiment with batch size of 1 without any other changes. The final result is 80.64%,  which is even lower than off-the-shelf deit-b's 82%. 

<!-- ![Image](https://github.com/user-attachments/assets/5e8f8a4d-8c20-4724-883d-a473faf64b0b) -->

![Image](img/bs1.png)

---

# SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers

Spatial Autocorrelation Token Analysis (SATA) enhances the robustness and efficiency of Vision Transformers (ViTs) without retraining. While previous efforts to improve ViTs relied on heavy training strategies, augmentation, or structural changes, SATA leverages spatial relationships between token features. By grouping tokens based on spatial autocorrelation scores before the Feed-Forward Network (FFN) block, it boosts both representational power and computational efficiency. SATA integrates seamlessly with pre-trained ViTs and achieves state-of-the-art results on **ImageNet-1K (94.9% top-1)** and robustness benchmarks like **ImageNet-A (63.6%)**, **ImageNet-R (79.2%)**, and **ImageNet-C (mCE 13.6%)**— **all without additional training or fine-tuning**.


## SATA pipeline
### Geographical Spatial Auto-correlation

![image](https://github.com/user-attachments/assets/19338553-3faa-4494-9c80-60377ca0645b)

![sata-pipeline](https://github.com/user-attachments/assets/6123c376-9dde-4819-96ad-d0a8a5c7df70)

### ViTs' Robustness Performance
![robust](https://github.com/user-attachments/assets/18b6d080-f71c-4b87-a8f0-d7482d9572e6)

### ImageNet-1k Classification Performance
![imagenet-1k](https://github.com/user-attachments/assets/67f71566-7eb0-4c0e-89ff-5ebc788338a2)

### Run
1- Ensure that all essential libraries and packages (pytorch, timm, and thop) are installed.

2- Run the **main-val.py** file using the following commands and settings:

```bash
python main-eval.py --model_name "deit_base_patch16_224" --gamma 0.7 --data_path ./ImageNet2012/val/ --sata 
```

### References
1. Nikzad, Nick, Yi Liao,Yongsheng Gao, and Jun Zhou. "SATA: Spatial Autocorrelation Token Analysis for Enhancing the Robustness of Vision Transformers", Accepted by Computer Vision and Pattern Recognition Conference (CVPR). 2025.
2. Nikzad, Nick, Yongsheng Gao, and Jun Zhou. ["CSA-Net: Channel-wise Spatially Autocorrelated Attention Networks."](https://arxiv.org/abs/2405.05755) arXiv preprint arXiv:2405.05755. 2024.
