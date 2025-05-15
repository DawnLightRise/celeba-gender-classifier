This repository provides a gender classification model trained on the publicly available [CelebA dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/data). It includes one summary file (1), three scripts (2–4), and a refined model (5):

1. [SUMMARY](SUMMARY)  
2. [celeba_gender_classifier.ipynb](celeba_gender_classifier.ipynb)  
3. [celeba_gender_classifier.py](celeba_gender_classifier.py)  
4. [run.sh](run.sh)  # This is an example of how to run the Python file.  
5. [best_model.pth](best_model.pth)  # This is the refined model. The backbone uses the pre-trained MobileNetV2 (ImageNet).

If you would like to review the results, you can view the [celeba_gender_classifier.ipynb](celeba_gender_classifier.ipynb) file.  
If you would like to test the Python code, you can run
```
bash run.sh
```
in a Linux terminal.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
