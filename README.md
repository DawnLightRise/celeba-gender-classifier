This repository includes four files (1-4) and a model (5):
1. SUMMARY
2. celeba_gender_classifier.ipynb
3. celeba_gender_classifier.py
4. run.sh  # This is an example of how to run the Python file.
5. best_model.pth # This is the refined model.

If you would like to review the results, you can view the celeba_gender_classifier.ipynb file.
If you would like to test the Python code, you can run `bash run.sh` in a Linux terminal.

## SUMMARY

Model & Training Procedure
Backbone: Pre-trained MobileNetV2 (ImageNet).
Head replacement: Final 1000-way classifier swapped for a 2-way linear layer.
Two-phase training:
- Warm-up (3 epochs): Freeze all backbone layers, train only the new head.
- Fine-tuning (7 epochs): Unfreeze entire network, train full model.

=== Warmup Phase ===
Warmup 1/3 • Loss 0.2662 • Acc 0.9145 • F1 0.8983
Warmup 2/3 • Loss 0.2593 • Acc 0.9155 • F1 0.9028
Warmup 3/3 • Loss 0.2596 • Acc 0.9163 • F1 0.9012

=== Fine-tune Phase ===
Finetune 1/7 • Loss 0.0792 • Acc 0.9862 • F1 0.9838
Finetune 2/7 • Loss 0.0430 • Acc 0.9858 • F1 0.9833
Finetune 3/7 • Loss 0.0303 • Acc 0.9873 • F1 0.9851
Finetune 4/7 • Loss 0.0205 • Acc 0.9863 • F1 0.9838
Finetune 5/7 • Loss 0.0149 • Acc 0.9857 • F1 0.9832
Finetune 6/7 • Loss 0.0113 • Acc 0.9864 • F1 0.9839
Finetune 7/7 • Loss 0.0091 • Acc 0.9875 • F1 0.9853

=== Test Set Metrics ===
Accuracy : 0.9811
Precision: 0.9819
Recall   : 0.9689
F1-score : 0.9753
