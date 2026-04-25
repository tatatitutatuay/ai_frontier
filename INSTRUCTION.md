Mini-Project หัวข้อ Biometrics

## Project Focus

Anti-spoofing: ศึกษาและพัฒนาโมเดลตรวจจับเสียงปลอม โดยโฟกัสที่ ThaiSpoof Dataset จาก AI For Thai เป็นหลัก

เป้าหมายของโปรเจกต์นี้คือสร้างระบบ binary classification สำหรับแยกเสียงจริง (genuine / bona fide) และเสียงปลอม (spoof / synthetic speech) โดยใช้ข้อมูลบางส่วนเท่านั้น เพราะ dataset มีขนาดใหญ่ และต้องให้สามารถทดลองได้จริงบนเครื่องส่วนตัวหรือ Google Colab

## Dataset Choice

Primary dataset:
- ThaiSpoof Dataset จาก AI For Thai

Do not use the full dataset. ให้เลือก subset ขนาดพอดีกับทรัพยากรเครื่อง:
- Train: ประมาณ 3,000 genuine + 3,000 spoof
- Test: ประมาณ 1,500 genuine + 1,500 spoof
- ถ้าเครื่องหรือ Colab memory ไม่พอ ให้ลดเหลือ 1,000 + 1,000 สำหรับ train และ 500 + 500 สำหรับ test

Optional backup dataset only if ThaiSpoof access is blocked:
- ASVspoof 2019 Logical Access (LA)

## Compute Constraints

The work must be practical on:
- MacBook Air M4 with 16 GB RAM
- Google Colab free or standard runtime

Constraints:
- Avoid training on the full dataset.
- Avoid very large architectures unless running on Colab GPU.
- Prefer cached feature extraction so audio does not need to be processed repeatedly.
- Keep batch size small, usually 8 or 16.
- Use early stopping to reduce training time.
- Save feature files and model checkpoints.
- Keep file paths configurable instead of hard-coding Windows, cluster, or local machine paths.

## Recommended Technical Plan

1. Prepare ThaiSpoof audio folders into two classes:
   - genuine
   - spoof

2. Create a balanced subset:
   - Train genuine
   - Train spoof
   - Test genuine
   - Test spoof

3. Extract acoustic features:
   - Primary feature: LFCC
   - Optional comparison feature: MFCC

4. Train baseline model:
   - LFCC + LCNN / small CNN

5. Train improvement model:
   - LFCC + lightweight ResNet-style CNN, or
   - MFCC vs LFCC comparison using the same model architecture

6. Evaluate with:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - EER (Equal Error Rate)
   - Confusion matrix

7. Compare results in a table:

| Experiment | Feature | Model | Accuracy | F1-score | EER |
| --- | --- | --- | --- | --- | --- |
| Baseline | LFCC | LCNN / small CNN | | | |
| Improvement | LFCC | lightweight ResNet / improved CNN | | | |
| Optional | MFCC | same model as baseline | | | |

## Existing Workspace Files

Current useful files:
- `ThaiSpoof/extract_lfcc.py`
- `ThaiSpoof/extract_mfcc.py`
- `ThaiSpoof/LFCC_pipeline.py`
- `ThaiSpoof/MFCC_pipeline.py`
- `ThaiSpoof/lcnn.py`
- `ThaiSpoof/resnet.py`
- `ThaiSpoof/lfcc_resnet_set.py`

Before running experiments, update scripts so dataset paths, output paths, sample counts, feature type, batch size, and epochs can be changed from config variables or command-line arguments.

## Report Direction

The final report should explain:
- Why voice anti-spoofing matters for biometric systems
- Why ThaiSpoof was selected
- Why only a subset was used
- How genuine/spoof samples were balanced
- How LFCC and/or MFCC features were extracted
- Baseline model architecture
- Improvement approach
- Evaluation metrics and results
- Limitations from dataset size and compute budget

## Original Reference Resources

https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset
https://www.asvspoof.org/
https://github.com/asvspoof
https://github.com/SLSCU/CSS

ตัวอย่างไอเดียสำหรับ improvement ดูได้จาก

https://ieeexplore.ieee.org/abstract/document/10354096
https://link.springer.com/chapter/10.1007/978-3-031-46781-3_13
https://ieeexplore.ieee.org/abstract/document/10665383
https://link.springer.com/chapter/10.1007/978-981-96-4606-7_13
https://link.springer.com/chapter/10.1007/978-981-96-4606-7_14
