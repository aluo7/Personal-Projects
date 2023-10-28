# PenaltyProphet
Utilizing convolutional neural network DenseNet121 inorder to predict where penalty kickers direct ball and where goalies dive to save penalty kick

## Table of Contents
1. [TLDR](#tldr)
2. [Introduction](#introduction)
3. [Approach](#approach)
4. [Results](#results)
5. [Potential Issues](#potential-issues)
6. [Future Plans](#future-plans)
7. [Dataset](#dataset)

## TLDR

Create a convolutional neural network by transfer learning of [imagenet](https://www.image-net.org/) with DenseNet121 while reducing overfitting techniques with batch normalization, dropout techniques, and image augmentation. PenaltyProphet model is able to predict 98% of penalty kickers' direction before striking the ball and 91% of goalies' direction of dive to prevent penalty kick

## Introduction

After the World Cup, there were many penalty shootouts that lead to exciting and heartbreaking moments. Being an ex-athlete myself, I understand how stressful it is to take a penalty kick having the whole world and your fans watching. Taking these penalty kicks can be difficult having many fans creating noise, being in a famous tournament such as the World Cup, and sometimes having all players take penalty kicks including the goalies who don't regularly practice penalty kicks.

Having this in mind, it inspired me to see how well convolutional neural networks would be in predicting where shooters will kick the penalty and where the goalie would dive inorder to save the penalty kick. This would give some information on what sort of techniques professional players use to score such great penalties and what gives their penalties away to the goalie and the viewers 

## Approach

Collecting data was quite annoying and tricky. Inorder to grab data for this project, I had to scrape youtube videos of professional penalty shoouts where the camera is behind the penalty Kicker. Downloading the video and manually seperating the frames of the video into 3 classifications (Left,Center,Right) indicating where the player shot the penalty. What is important here is that frames of the long run up to the ball is not in the dataset because soccer players already know where they are going to kick before the penalty shootout even starts. The few frames of when they go up to strike the ball is what is important and where technique comes out. 
![Example of images being loaded into classification](https://i.ytimg.com/vi/_TFribViDSs/maxresdefault.jpg)
*Example of camera position being loaded into classification*

Right and left penalty kicks had ~800 images and and center images ~300 images due to taking a center penalty kick being "very risky" in the penalty kicks since that is where the goalie typically stands before kick is taken. Since this is image classification problem, we would need to use a convultional neural network inorder to predict the images of when the player comes up to take the penalty kick. Instead of trying to start from scratch, I decided to use transfer learning from imagenet to incorporate within DenseNet121 inorder to get weights that are pretrained on recognizing shapes and patterns.

Since the dataset we are working with is very small, I implemented some overfitting reduction techniques within the DenseNet121 model:
- 2 learnable dense layers with dropout probability of .5
- Batchnormalization with two dense layers
- Image augmentation sequence
    - Horizontal Flip (p=.5)
    - Veritcal Flip (p=.5)
    - Rotation (p=.6,limit=22°)
    - Coarse Dropout (p=.5)
    - Weather Augmentation (p=.5)
    - Color Augmentation (p=.5)
- 70-15-15 datasplit that tends to work well for small datasets

The same technique and images were used to train a separate model that predicts where the goal keeper will dive before the penalty kick was taken

## Results

After running the penalty kicker model for 10 epochs, model was able to produce results of 98% accuracy.
After running the goalie keeper model for 5 epochs, model was able to produce results of 91% accuracy

## Potential Issues

1. Small dataset due to time consumption of gathering data
2. Potential bias could be present within the data between left and right footed players. Since there are typically more right footed players than left footed players, it is hard to incorporate left footed players within the model especially scraping frames from Youtube, limit camera penalty kicks from this angle (Though this will change in the future due to in-air cameras now in most stadiums), and limited left footed players
3. Potentially could not generalize well even after all overfitting reduction techniques

## Future Plans
1. Grab more data not only from camera angle (provided from before) but camera angle that is in the stadium
![Example of images being loaded into classification](https://i.kinja-img.com/gawker-media/image/upload/c_fill,f_auto,fl_progressive,g_center,h_675,pg_1,q_80,w_1200/f4b3f333f9a497d1c4d75df6b48403a6.jpg)
*Example of camera angle from the stadium*

2. Grab equal datasets of both left and right footed players inorder to remove bias
3. Try Strat K-Fold inorder to see if accuracy improves or generalization

## Dataset

Dataset used to train model: https://drive.google.com/drive/folders/1ofFvs8fq3BYuj0YE7H2Bzv3fqdgncHFM?usp=sharing
