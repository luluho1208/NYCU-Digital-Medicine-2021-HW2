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
The dataset we used is the subset from [kaggle dataset: SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection). The orginial data has 6054 training data and are available for object detection task. In this project, we only use 1200 dicom file as train data (including 400 Negative, 400 Typical and 400 Atypical examples). We use Cross-Validation to test our performance, so we create the kfold.txt in dcm_folds

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

The model we used in this project is [Vision Transformer (ViT)](https://github.com/google-research/vision_transformer). This model is developed by Google and they release some pretrain weight for easily classification tasks. We use pretrain weight : ViT_B-16, which is powerful and pretrained on ImageNet21k and use transfer learning in this project.

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
        
    It is highly recommanded to have at least 2 GPU to run the experiment.
    
In "requirements.txt", we have:

    opencv-python==4.5.4.60
    Pillow==8.4.0
    tqdm==4.62.3
    scikit-learn==0.24.2
    torch==1.10.0
    torchvision==0.11.1
    ml_collections==0.1.0
    googledrivedownloader==0.4
    requests==2.26.0
    wget==3.2

Usage
---
1. Git clone this project to local
```git=
git clone https://github.com/luluho1208/NYCU-Digital-Medicine-2021-HW2.git
```

2. Use the virtualenv to create the virtual environment:
```bash=
cd NYCU-Digital-Medicine-2021-HW2
virtualenv -p /usr/bin/python3 myenv
source myenv/bin/activate
pip install -r requirements.txt
```

3. Download the pretrained ViT model weight and training/test dataset
```git=
python setup.py
```

4. Start training your model
```git=
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --kfold=0 --epoch=20 --lr=0.001 --resampling=3 --patience=3 --batch_size=4 --num_classes=3 --image_size=528 --root_dir=./ --saved_model_path=./save_model/ViT_1124.pth
```

5. Use the saved model to generate submission.csv file
```git=
CUDA_VISIBLE_DEVICES=0 python inference.py --batch_size=2 --num_classes=3 --image_size=528 --root_dir=./ --pthfile=./save_model/ViT_1124.pth
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
