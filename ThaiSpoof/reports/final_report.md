# Voice Anti-Spoofing with ThaiSpoof

## 1. Introduction

Voice biometric systems are vulnerable to spoofing attacks from synthetic speech, voice conversion, and other generated audio. This project studies voice anti-spoofing as a binary classification problem: classify each utterance as either genuine speech or spoofed speech.

The goal is to build a practical experiment pipeline that can run on a MacBook Air M4 with 16 GB RAM while still producing results suitable for comparison in a biometrics mini-project.

## 2. Dataset

The project uses ThaiSpoof-style Thai speech data from the downloaded local folders under `data/raw/`:

- `data/raw/genuine/` for bona fide human speech
- `data/raw/Corpus-Spoof-VAJA/` for spoofed speech

The full available local dataset contains 4,583 genuine files and 4,583 spoof files. A balanced subset was used to keep training practical on local CPU hardware.

| Split | Genuine | Spoof | Total |
| --- | ---: | ---: | ---: |
| Train | 800 | 800 | 1,600 |
| Validation | 200 | 200 | 400 |
| Test | 500 | 500 | 1,000 |

The train and validation rows come from a 1,000 genuine + 1,000 spoof training subset with a 20% validation split. The test set uses 500 genuine + 500 spoof files.

## 3. Feature Extraction

Two cepstral feature types were compared:

- **LFCC**: linear-frequency cepstral coefficients. LFCC is commonly used in anti-spoofing research because linear-frequency spacing can preserve high-frequency artifacts that help reveal synthetic speech.
- **MFCC**: mel-frequency cepstral coefficients. MFCC is a common speech-processing baseline and was used as an optional comparison.

Each audio file was converted into a fixed-size feature matrix. Short utterances were repeated to reach the target frame length, while long utterances were truncated.

## 4. Models

Three experiments were run:

| Experiment | Feature | Model | Purpose |
| --- | --- | --- | --- |
| Baseline | LFCC | `small_cnn` | Main lightweight baseline |
| Improvement | LFCC | `resnet_lite` | Deeper residual CNN comparison |
| Optional | MFCC | `small_cnn` | Feature comparison against LFCC |

The models were trained with small batch sizes and early-stopping support to fit the compute budget.

## 5. Evaluation Metrics

The project reports:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- EER, or Equal Error Rate
- Confusion matrix counts

Spoof speech is treated as the positive class.

## 6. Results

The table below reports the **test split** results.

| Experiment | Feature | Model | Accuracy | Precision | Recall | F1-score | EER |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | LFCC | `small_cnn` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Improvement | LFCC | `resnet_lite` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| Optional | MFCC | `small_cnn` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

Confusion matrix counts on the test split:

| Experiment | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| LFCC + `small_cnn` | 500 | 0 | 0 | 500 |
| LFCC + `resnet_lite` | 500 | 0 | 0 | 500 |
| MFCC + `small_cnn` | 500 | 0 | 0 | 500 |

## 7. Discussion

All three experiments achieved perfect test-set performance on the selected balanced subset. This means the models separated the downloaded genuine and spoof files without any observed classification errors.

The result is strong, but it should be interpreted carefully. Perfect performance can happen when spoofed speech has clear artifacts, but it can also indicate that the classifier is learning dataset-source differences rather than general spoofing cues. For example, genuine files and spoof files may differ in generation pipeline, file structure, acoustic channel, duration distribution, or other metadata-related artifacts.

The LFCC baseline and ResNet-lite improvement produced the same final test metrics. Because the lightweight CNN already reached perfect accuracy and EER, the deeper residual model did not show measurable improvement on this subset. MFCC also reached the same score, suggesting that this local subset may be easy to separate with standard cepstral features.

## 8. Limitations

- The full dataset was not used; experiments used a balanced subset suitable for a MacBook Air M4.
- The test split comes from the same downloaded data source as the training split, so it may not measure cross-dataset generalization.
- Perfect scores may reflect source or synthesis artifacts specific to this dataset subset.
- Fixed-size feature matrices truncate long utterances and repeat short utterances.
- Training used local CPU-friendly model sizes rather than large anti-spoofing architectures.

## 9. Conclusion

The project successfully built a practical Thai voice anti-spoofing pipeline using balanced genuine and spoof subsets, cached acoustic feature extraction, and lightweight CNN-based classifiers.

On the selected Mac-sized subset, all tested configurations reached 100% test accuracy, 1.000 F1-score, and 0.000 EER. The best practical choice is therefore the LFCC + `small_cnn` baseline because it achieved the same performance as the deeper model with lower complexity.

Future work should test on a larger and more diverse split, evaluate cross-source generalization, and include additional spoof types to check whether the perfect subset performance remains robust.
