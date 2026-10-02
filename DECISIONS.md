# Design Decisions

Step-by-step record of methodological decisions for the agricultural pest image classification pipeline, with the rationale for each.
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
- **Features:** HOG (shape: histogram of gradient orientations per cell) and LBP (texture: histogram of local binary codes), computed on grayscale images and concatenated into one vector per image (1,776 columns). Color features are excluded, consistent with the exploratory analysis.
  - **HOG:** 32 x 32 pixel cells, 2 x 2 blocks, 9 orientations (1,296 columns). A first version with 16 x 16 cells (6,084 columns) made HOG about 97% of the vector, so it dominated the SVM distance, and gave more columns than training samples.
  - **LBP:** 8 neighbors, radii 1, 2 and 3 (uniform, 10 codes each), histogram over a 4 x 4 grid (3 x 160 = 480 columns).
  - **Effect (validation, first version to current):** SVM macro F1 0.334 to 0.378; Random Forest 0.294 to 0.332. The gain was not separated between the HOG and the LBP changes.
- **Classifiers:** SVM (RBF kernel, default parameters) and Random Forest (300 trees). Feature standardization is fitted on the training set only. Parallelism limited to 6 jobs.
- **Hyperparameters:** not tuned. Default parameters for the SVM and 300 trees for the Random Forest.
- **Ablation:** not performed (HOG only, LBP only). The features are used concatenated.
- **Metrics:** accuracy, macro F1, confusion matrix on validation (SVM: accuracy 0.411, macro F1 0.378; Random Forest: 0.379 and 0.332). The SVM is the baseline for comparison with deep learning. The test set is evaluated once, in the final comparison.
- **Error pattern (SVM, validation, values read from the confusion matrix):** best classes are bees and moth; worst are beetle (2 of about 62), slug (8 of about 59), earwig and caterpillar. Main confusions: wasp as bees, beetle as grasshopper, slug as earthworms and snail. Beetle is not a small class, so the errors come from visual similarity (shape and background), not from class size.

## 6. Data augmentation (deep learning only)
- **Online** augmentation, applied to the training set only, with Albumentations. Validation and test images are not augmented. Online generation avoids storing augmented copies and shows new variations at every epoch (Krizhevsky et al., 2012, generated augmented images on the CPU during training, without storing them).
- **Core transformations (p = 0.5 each):** horizontal flip; rotation of up to 30 degrees, corners filled with the letterbox padding color; random crop covering at least 70% of the image (zoom in only); brightness and contrast jitter of 0.2.
- **Degradations (one of Gaussian blur, Gaussian noise or JPEG compression, in 20% of the images):** applied less often because they remove the texture that separates similar classes.
- **Not included:** perspective distortion, which may deform the shape that separates the classes.
- **Rationale and limits:** the values of p and the intensities are design choices justified by the dataset characteristics (insects have no fixed orientation; luminance differs between classes; the insect must stay in the frame), not values taken from the literature or tuned. p = 0.5 is a neutral setting; 0.2 for the degradations limits the loss of detail. The effect is checked by comparing the models with and without augmentation (Section 7).
- **Class imbalance (1.55x):** class weights in the loss (`balanced`: n_samples / (n_classes x class_samples)), between about 0.9 and 1.4. Chosen for simplicity and because the imbalance is mild. Class weights do not address visual similarity between classes.
- **References checked:** Krizhevsky et al. (2012); Perez and Wang (2017, arXiv:1712.04621); Buslaev et al. (2018, arXiv:1809.06839). Shorten and Khoshgoftaar (2019, J. Big Data 6) was not read in full, so no claim is attributed to it.

## 7. Deep learning approach
- **Framework:** Keras 3 with the PyTorch backend (GPU training with a single 12 GB GPU). The backend is set before importing Keras.
- **Models:**
  - A small CNN trained from scratch: a first convolution with stride 2 (32 filters) followed by 3 blocks of 2 convolutions (64, 128, 256 filters), each with batch normalization and max pooling, then global average pooling and dropout 0.4. Pixels are rescaled to [0, 1] inside the model. The stride-2 first convolution reduces memory use (a version with 4 blocks of 2 convolutions at full resolution used the whole 12 GB of GPU memory).
  - MobileNetV2 with ImageNet weights (transfer learning), pixels rescaled to [-1, 1] (same as its `preprocess_input`), global average pooling, dropout 0.3. First the feature extractor is frozen and only the classifier is trained; then the best configuration is fine-tuned (last 30 layers unfrozen, batch normalization layers kept frozen, learning rate 10x smaller).
- **Memory and speed:** mixed precision (`mixed_float16`, output layer in float32); the last incomplete batch is dropped (3,841 images = 60 batches of 64 + 1); garbage collection and `torch.cuda.empty_cache()` after every epoch, because without it the GPU memory in use grew by about 1 GB per epoch and the training slowed down. With these changes the CNN runs at about 4 s per epoch with a peak of about 1.5 GB of GPU memory (measured on the full training set). Predictions use batches of 64.
- **Training:** online augmentation through a `PyDataset` (2 CPU threads); class weights passed as sample weights; sparse categorical cross-entropy; `ReduceLROnPlateau` (factor 0.5, patience 3) and early stopping (patience 8, best weights restored) on the validation loss.
- **Hyperparameter experiment (MobileNetV2, frozen):** (Adam, 1e-3, batch 64), (Adam, 1e-4, 64), (SGD with momentum 0.9, 1e-2, 64), (Adam, 1e-3, 32). The best one by validation macro F1 is the one fine-tuned. The CNN from scratch uses a single configuration (Adam, 1e-3, 64).
- **Evaluation:** accuracy, macro precision, macro recall and macro F1 on train, validation and test for every model (traditional and deep learning), shown in one table (with a short legend of the metrics) after each training and saved in `results_all_models.csv`. Macro averages give the same weight to each of the 12 classes, which suits the mild class imbalance. The test numbers are reported for all models, but the models are chosen using validation only. Comparison with the traditional baseline; qualitative error analysis of the best deep learning model on the validation set and, for the final report, on the test set (classification report, confusion matrix, 12 misclassified and 12 correctly classified images, with the model name and the number of correct and wrong predictions in the title). The test predictions are computed from the saved file of the best model.
- **Results of the first full run (macro F1, validation / test):** SVM 0.378 / 0.371 (baseline); Random Forest 0.332 / 0.318; CNN from scratch 0.555 / 0.574 (it did not converge within 40 epochs); MobileNetV2 frozen 0.853 to 0.882 / 0.834 to 0.863 depending on the configuration; **MobileNetV2 fine-tuned 0.908 / 0.882 (best)**. The training is not exactly repeatable (augmentation threads and GPU), so the numbers can vary slightly between runs. The configurations of the frozen MobileNetV2 that reached the best validation F1 differ by less than 0.5 points, which is within the noise of a single run.
- **Ablation:** not performed. The effect of augmentation was not measured separately. Training from scratch and transfer learning are compared through the two models.

## 8. Reproducibility
- **Seeds:** fixed random seeds (split seed 14, Keras seed). The training of the deep learning models is not exactly repeatable, so small differences between runs are expected.
- **Dataset path:** relative to the notebook. The folder `Agricultural Pests Image Dataset/` is expected next to the notebook, or another folder can be given through the environment variable `PEST_DATA_DIR`. There are no absolute paths in the code, and the cell fails with a clear message if the folder is missing. The dataset is delivered together with the notebook.
- **Generated files:** everything the notebook creates is written to the working folder, which is the notebook folder: the split table (`df_clean_crop_pests.csv`), the resized images (`images_224_rgb.npy`), the scaled features (`features_traditional.npz`), the results table (`results_all_models.csv`) and the folder `models/`. The notebook does not read these files back, except the results table, so it can run from scratch in a clean folder (the results table must be removed first, otherwise old rows are kept).
- **Saved models:** deep learning models as `.keras` and the SVM, the Random Forest and the scaler as `.joblib`, all in `models/`, so that the results can be generated locally and inspected later.
- **Version control:** `.gitignore` excludes `models/`, `*.npy`, `*.npz`, the split table and the dataset folder; the results table is versioned.
- **Notebook organization:** it follows the order of the report template (Sections 1 to 8, Digest and References). All imports are in the first code cell, grouped in blocks. The Keras backend (`torch`) is set in that cell before `keras` is imported.
- **Execution:** the notebook is executed from start to end in a clean folder before export. The deep learning part takes about 20 minutes on a 12 GB GPU.

## References
- Buslaev, A., Parinov, A., Khvedchenya, E., Iglovikov, V. I. and Kalinin, A. A. (2018). *Albumentations: fast and flexible image augmentations.* arXiv:1809.06839.
- Goodfellow, I., Bengio, Y. and Courville, A. (2016). *Deep Learning.* MIT Press.
- Krizhevsky, A., Sutskever, I. and Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* Advances in Neural Information Processing Systems 25.
- Perez, L. and Wang, J. (2017). *The Effectiveness of Data Augmentation in Image Classification using Deep Learning.* arXiv:1712.04621.
- Joseph, V. R. (2022). *Optimal ratio for data splitting.* Statistical Analysis and Data Mining, 15, 531–538.
