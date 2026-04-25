# Voice Anti-Spoofing with ThaiSpoof

## 1. Introduction

Voice biometrics can be attacked with synthetic speech or transformed speech. This project studies anti-spoofing as a binary classification problem: genuine speech vs spoofed speech.

## 2. Dataset

Dataset: ThaiSpoof from AI For Thai.

Only a subset was used because the full dataset is too large for a MacBook Air M4 with 16 GB RAM and is also time-consuming on Google Colab.

Subset:

| Split | Genuine | Spoof |
| --- | ---: | ---: |
| Train | | |
| Validation | | |
| Test | | |

## 3. Feature Extraction

Primary feature: LFCC.

LFCC was selected because linear-frequency cepstral features are commonly used in speech spoofing countermeasures and preserve frequency-domain artifacts that may be useful for detecting synthetic speech.

Optional comparison: MFCC.

## 4. Models

Baseline: small CNN trained on LFCC features.

Improvement: lightweight ResNet-style CNN trained on the same LFCC features.

Both models use early stopping and small batch sizes so experiments fit the available compute budget.

## 5. Evaluation Metrics

The project reports:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1-score
- EER (Equal Error Rate)
- Confusion matrix counts

## 6. Results

| Experiment | Feature | Model | Accuracy | Precision | Recall | F1-score | EER |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | LFCC | small_cnn | | | | | |
| Improvement | LFCC | resnet_lite | | | | | |
| Optional | MFCC | small_cnn | | | | | |

## 7. Discussion

Discuss whether the improvement model reduced EER or improved F1-score compared with the baseline. If results are close, discuss whether the smaller dataset, attack diversity, or training time may have limited model performance.

## 8. Limitations

- The full ThaiSpoof dataset was not used.
- The experiment used fixed-size feature matrices, so long utterances are truncated and short utterances are repeated.
- Mac runs use CPU training, which limits architecture size and epoch count.
- Results may vary depending on which spoof types are sampled.

## 9. Conclusion

Summarize the best model, best feature setting, and the main lesson from the anti-spoofing experiment.
