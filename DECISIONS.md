# Design Decisions

Step-by-step record of methodological decisions for the agricultural pest image classification pipeline, with the rationale for each. Items marked *pending* are not yet decided.

## 1. Task and dataset
- **Task:** multiclass image classification (12 classes: ants, bees, beetle, caterpillar, earthworms, earwig, grasshopper, moth, slug, snail, wasp, weevil).
- **Dataset:** Agricultural Pests Image Dataset (Kaggle): https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset
- **Rationale:** pest categories affect multiple crops, unlike plant-disease datasets, which are usually specific to one plant and differ in subtle visual detail.
- **Archival:** a local copy of the dataset is kept, since the hosted version may be removed.
- **Subsampling:** not applied. The dataset is small (5,494 images, low resolution).

## 2. Exploratory analysis: findings and consequences

| Finding | Consequence |
|---|---|
| Mild class imbalance (1.55x; 323 to 500 images per class) | Stratified split; macro-averaged F1 |
| 6 exact duplicate groups (MD5 hash), none across classes | Remove duplicates before splitting (5,488 images remain) |
| Image sizes: maximum side of 300 px (5,477 images) or 512 px (11); aspect ratio 0.31 to 3.41 | Aspect-ratio-preserving resize with padding |
| Mean color histograms are nearly identical across classes (background dominates) | Color is a weak feature; favor shape and texture |
| Class-dependent luminance (about 26 points between the brightest and darkest classes) | Brightness/contrast jitter in augmentation, to avoid shortcut learning |
| Sharpness is right-skewed; low-contrast and dark images still show the insect | No sharpening or contrast enhancement |
| No visible noise | No denoising |

## 3. Data splitting
- **Split:** 70% train / 15% validation / 15% test (3,841 / 823 / 824 images), stratified by class, fixed seed (14), applied after duplicate removal.
- **Roles:** train to fit; validation for hyperparameter tuning and early stopping; test used once for the final evaluation.
- **Ratio rationale:** a larger training share (about 80%) is a common convention (Goodfellow et al., 2016), but the smallest class (earthworms, 323 images) would have about 32 images in each of the validation and test sets. With 15%, it has 48 to 49, which makes per-class evaluation more reliable. The cost is about 550 fewer training images.
- **Limitation:** no split ratio is universally optimal (Joseph, 2022). The choice of 70/15/15 was not compared empirically against alternatives.
- **Reproducibility:** the split is stored as a column of the cleaned metadata table and exported to CSV.

## 4. Preprocessing
- **Resize:** letterbox to 224 x 224. The image is scaled by its larger side (area interpolation) and centered; the aspect ratio is preserved because object shape is discriminative. All images are downscaled; none is upscaled.
- **Padding color:** per-channel mean color over all training images, pooled across classes. A class-specific color is rejected because it would leak the label into the input.
- **Format:** BGR to RGB conversion; storage as `uint8` (about 826 MB for the full dataset).
- **Not applied:** noise filtering (no noise observed), per-channel thresholding (backgrounds vary between images), contrast equalization (insects remain visible).
- **Normalization:**
  - Deep learning: pixel scaling with the preprocessing function of the chosen pretrained model.
  - SVM: feature standardization fitted on the training set only.
  - Random Forest: no scaling required.
  - HOG and LBP: no pixel normalization required.

## 5. Traditional approach
- **Features:** HOG (shape: histogram of gradient orientations per cell) and LBP (texture: histogram of local binary codes), computed on grayscale images and concatenated into one vector per image. Color features are excluded, consistent with the exploratory analysis.
- **Classifiers:** SVM (RBF kernel) and Random Forest.
- **Hyperparameters:** tuned on the validation set.
- **Ablation:** HOG only, LBP only, and HOG + LBP.
- **Metrics:** accuracy, macro F1, confusion matrix. The best configuration is the baseline for comparison with deep learning.
- **Optional extension, pending:** hue/saturation histogram (excluding value) as a third feature group, only if time allows.

## 6. Data augmentation (deep learning only)
- Online augmentation, applied to the training set only.
- Includes brightness/contrast jitter (see Section 2) plus geometric and photometric transformations, each justified by the dataset characteristics.
- Targeted handling of class imbalance: *pending*.

## 7. Deep learning approach
- **Framework:** Keras 3 with the PyTorch backend (GPU training with a single 12 GB GPU). The backend is set before importing Keras.
- **Models:** a small CNN trained from scratch and a pretrained MobileNetV2 (transfer learning).
- **Training:** optimizer, learning rate and batch size tuned on validation; `ReduceLROnPlateau` and early stopping on validation.
- **Evaluation:** one evaluation on the test set; comparison with the traditional baseline; qualitative error analysis (misclassified examples, confused class pairs).
- **Ablation, if time allows:** with versus without augmentation; training from scratch versus transfer learning.

## 8. Reproducibility
- Fixed random seeds; stored split and extracted features.
- The dataset path is configurable through a single variable.
- The notebook is executed from start to end before export.

## References
- Goodfellow, I., Bengio, Y. and Courville, A. (2016). *Deep Learning.* MIT Press.
- Joseph, V. R. (2022). *Optimal ratio for data splitting.* Statistical Analysis and Data Mining, 15, 531–538.
