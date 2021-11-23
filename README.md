Overview
---
The Task is to classify the given CXR (Chest X- Ray) images and tell whether the patient is normal, pneumonia, or comfirmed from the covid-19.

![](https://i.imgur.com/iVcEz32.png)

Our pipeline
---
![](https://i.imgur.com/6uwduWv.png)


![](https://i.imgur.com/FeTaaLY.png)


Dataset
---
The dataset we used is the subset from kaggle dataset: SIIM-FISABIO-RSNA COVID-19 Detection (!TODO 超連結https://www.kaggle.com/c/siim-covid19-detection). The orginial data has 6054 training data and are available for object detection task. In this project, we only use 1200 dicom file as train data (including 400 Negative, 400 Typical and 400 Atypical examples). We use Cross-Validation to test our performance, so we create the kfold.txt in TODO!!!!

![](https://i.imgur.com/dNSaffV.png)

![](https://i.imgur.com/J9GD7FZ.png)


Data Pre-processing
---
In this phase, we first translate the dicom file to png file with size 1024 with fixed-ratio. And we use the torch.transforms to do the augmentation. We use RandomAffile, RandomCrop, Resize, RandomHorizontalFlip, and Normalize.

For better performance, we resize the image to 528*528, which is a little large than used model training. Because the data is CXR, we need to do some data enhencement like Histogram Equalization or CLAHE to increase the contrast value in the image, so we can see some detail which is important for model training.


Also, We enlarge our training data by resampling 3 times the original data, and we split 20% to be the validation set. Finally, we have 2880 training data and 240 validation data. We utilize the torch customDataset and Dataloader to wrap our data.


![](https://i.imgur.com/IxVBTP6.png)

Model and Hyper-parameter
---

The model we used in this project is Vision Transformer (ViT) [TODO 超連結] ![](https://i.imgur.com/oJjagNv.png). This model is developed by Google and they release some pretrain weight for easily classification tasks. We use pretrain weight : ViT_B-16, which is powerful and pretrained on ImageNet21k and use transfer learning in this project.

* lr : 0.005
* epoch : 20
* patience : 3
* resampling : 3
* batch_size : 16
* image_size : 528
* Loss function: CrossEntropyLoss
* Optimizer : SGD(model.parameters(), momentum=0.9, weight_decay=1e-2)
* Scheduler : ReduceLROnPlateau(optimizer, factor=0.1, patience=3)

![](https://i.imgur.com/tq7HXY9.png)

Benchmark
---
We have 3 labels in this project, and we use the f1-score to be our evaluation metric. We use the sklearn.f1_score (average: macro) to calculate our performance. We save the highest val f1_score epoch model (.pth) as our final model weight and use it to generate the submission.csv file.

Prerequisite
---
    OS:Linux

    environment:

        Python 3.6.9    
        torch == 1.10.0
        CUDA version == 10.0

    It is highly recommanded to have at least 1 GPU to run the experiment.

In "requirements.txt", we have:

    opencv-python==4.5.4.60
    Pillow==8.4.0
    tqdm==4.62.3
    scikit-learn==0.24.2
    torch==1.10.0
    torchvision==0.11.1
    ml_collections==0.1.0

Usage
---
git clone this page to the local
```git=
git clone https://github.com/s106062339/DM_case1.git
```

use the virtualenv to create the virtual environment:
```bash=
virtualenv -p /usr/bin/python3 myenv
source myenv/bin/activate
pip install -r requirements.txt
```

Then we can start to prepare the data by running "prepare_data.py" , you need to modify the variable 'Data_dir' to meet your Case Presentation 1 Data file.

In this file, we also need to set the variable 'Search_Len' as any value of 100, 300, 500, 1000, for example, to decide the length of text to be chosen from the original case file.

The new created data will be saved in ./Case Presentation 1 Data/{target}_{p,h}_{Search_Len}

So, if we want to run the experiment in Search_Len = 500, we will create 6 sub-folder under ./Case Presentation 1 Data:

    ./Train_p_500
    ./Train_h_500
    ./Test_p_500
    ./Test_h_500
    ./Validation_p_500
    ./Validation_h_500

:bulb:Run the command below to execute     
```bash=
python prepare_data.py
```

For the file "Bert_obesity_classifier.py":

You need to set the variable 'Data_dir' to meet yor Case Presentation 1 Data file.
Please name the variable 'csv_name', which is the output csv file name stored in ./Submission folder, and remember to set the variable 'Search_Len' to run the experiment.

If you encounter the CUDA Out of Memory problem, please try to half the batch size, and we can run "Bert_obesity_classifier.py" to train the model.
We will save the P and H model in ./model which can easy reproduce our result.

:bulb:Run the command below to execute     
```bash=
CUDA_VISIBLE_DEVICES=0 python Bert_obesity_classifier.py
```

Experiment Result:
---
Kaggle competition -- Public leaderboard

![](https://i.imgur.com/eRzgAVC.png)

Kaggle competition -- Private leaderboard

![](https://i.imgur.com/eRzgAVC.png)



Training processing of our ViT model:

![](https://i.imgur.com/AtKWdKJ.png)


validation processing of our ViT model:

![](https://i.imgur.com/5CjxVkf.png)

Learning Rate processing of our ViT model:

![](https://i.imgur.com/orN28kR.png)


We can see that although we get No.4 in public leaderboard, our model are not stable enough. Our validation loss can't converge successfully. I think the model is hard to train because the ViT model is powerful but we has limited data, which may cause overfitting easily. Also, The ViT is pretrained on ImageNet21k which is far from CXR type image data. So, we don't get higher performance and more stable training/validation processing.
