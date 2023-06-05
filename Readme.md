# Skin Cancer Experiment Log

## Description
 - Exploring the effects of class ordering on model performance to gain insight into optimal task ordering for devising a curriculum for continual learning
### 1. Dataset: 
 #### Name: ISIC 2019
 #### Classes: 8
 #### Hierarchy: 2 Levels
 #### Total Samples: 25,331
 #### Parent Classes: Melanocytic, Non-Melanocytic
 #### Child Classes: 
 1. Melanoma
 2. Melanocytic Nevus
 3. Basal Cell Carcinoma
 4. Actinic Keratosis
 5. Benign Keratosis
 6. Squamous Cell Carcinoma
 7. Vascular Lesion
 8. Dermatofibroma

    | Melanocytic | Number of Samples | Non-Melanocytic | Number of Samples |
    | :---:       | :--:              | :---:           | :--:              | 
    | 1. Melanocytic Nevus|12,875  | 1. Basal Cell Carcinoma| 3,323|
    | 2. Melanoma | 4,552|2. Benign Keratosis | 2,624|
    | - | - | 3. Actinic Keratosis| 867|
    | - | - |4. Squamous Cell Carcinoma |628 |
    | - | - |5. Vascular Lesion| 253 |
    | - | - |6. Dermatofibroma| 239 |




## Index
| **Experiment Name**| **Experiment-ID**      | **Aim** | **Hypothesis**     | **Reference File** | 
|:---: | :---:        |    :----:     |          :-----:     |    :---:       |
| Melanocytic Nevus vs Melanoma | [melanocytic_exp1](#melanocytic-nevus-vs-melanoma)    | *To ascertain the distinguishability between Melanoma and Melanocytic Nevus*        | As per literature, these two classes appear almost identical visually and evolution of the nevi is required for final classification   | melanocytic_exp1.ipynb| 
| Melanocytic Nevus vs Melanoma (balanced) | [melanocytic_exp2](#melanocytic-nevus-vs-melanoma-balanced)    | *To ascertain the distinguishability between Melanoma and Melanocytic Nevus with balanced number of samples per class*        | Melanoma and Melanocytic Nevus are hard to distinguish. We aim to see roughly 50% accuracy for the binary classification task with the above two classes after balancing each class.   | melanocytic_exp2.ipynb|
|Melanoma vs Dysplastic Nevi | [dysplastic_nevus_exp1](#melanoma-vs-dysplastic-nevi) | *To ascertain the distinguishibility between melanoma and dysplastic nevus* | Dysplastic nevus and splitz nevus are two classes under the parent class of Melanocytic Nevus which are the most similar to melanoma, we suspect these classes are negatively affecting the model performance due to similar appearance to melanoma. We want to determine if it is possible to distinguish melanoma from the above 2 classes and we wish to investigate if doing so yields any clinically relevant concepts.| dysplastic_nevus_exp1.ipynb|

## **Melanocytic Nevus vs Melanoma**
### **Aim**
 -  To ascertain the distinguishability between Melanoma and Melanocytic Nevus  
### **Procedure**
 - We train a binary classifier on Melanoma vs Melanocytic Nevus
 #### *Experiment Details*
  1. **Architecture:** ResNet101
  2. **Train Epochs:** 14 
  3. **Optimizer:** *SGD*, *lr*: 0.001, *momentum*: 0.9
  4. **Loss:** *CrossEntropyLoss*, unweighed
  5. **Additional:** 
   - Images are resized to 224x224 to meet ResNet Specifications
   - Model was retrained from scratch
### **Observation**
 - As per the hypothesis, poor accuracy was expected. However, the final test accuracy came out to be 79% on the stratified test set.
 - A closer inspection of the dataset reveals that the classes themselves have a ratio of $73.8:26.2$ (Melanocytic Nevus to Melanoma).
 - Thus, the baseline for comparision has to be 74% and not 50% as is usually done for binary classification.

 <center>
    <div class="row" style="display: flex;">
    <div class="col" style="flex: 50%; padding: 5px;">
    <figure id="fig1"> 
    <img src="./images/acc_loss_melanocytic_exp1.1.png">
    <figcaption><p align="center">Fig.1 Accuracy Loss Curve for MEL vs NEV</p></figcaption>
    </figure>
    </div>
    <div class="col" style="flex: 50%; padding: 5px;">
    <figure id="fig2"> 
    <img src="./images/acc_loss_melanocytic_exp1.2.png">
    <figcaption><p align="center">Fig.2 Accuracy Comparision with Balanced Baseline</p></figcaption>
    </figure>
    </div>
    </div>
 </center>

 - From [Figure 1](#fig1) We notice the accuracy jitters around 0.76. [Figure 2](#fig2) draws our attention to the fact that the accuracy obtained is indeed very poor when considering the class imbalance of the dataset. The accuracy is barely above the threshold!
 #### *Classise Accuracy*

 |Class|Accuracy|
 |:----|---:|
 |1. Melanocytic Nevus|94.3%|
 |2. Melanoma |34.1%|
 - The classwise accuracy seems to be skewed heavily in favour of Melanocytic nevus. However, it is not as if the model is making random predictions for either class.


### **Conclusion**
 - The above results provide circumstantial evidence of Melanoma not being distinguishable from Melanocytic nevus.
 - Although it provides a strong base for further experimentation on there distinguishing capability after balancing, it is possible that the model may have been motivated to predict Nevus due to imbalance. 
 - Thus, preliminary results indicate that these two classes are not sufficiently distinguishable from each other. Due to an imbalance in the number of sample, the predictions are skewed towards Nevus (high recall (94% accuracy)) in comparision to Melanoma (low recall (34% accuracy)).


## **Melanocytic Nevus vs Melanoma (balanced)**
### **Aim**
 -  To ascertain the distinguishability between Melanoma and Melanocytic Nevus with balanced number of samples per class 
### **Procedure**
 - Continuing from the previous Experiment, we explore the distinguishability of Melanoma and Melanocytic Nevus. This time the experiments are performed on a balanced dataset
 - We restrict number of samples in Melanocytic Nevus to the first 4,552 images.
 - Shuffling before sampling is not necessary since the order within the dataset is random.
 - We train a binary classifier on Melanoma vs Melanocytic Nevus.
 #### *Experiment Details*
  1. **Architecture:** ResNet101
  2. **Train Epochs:** 14 
  3. **Optimizer:** *SGD*, *lr*: 0.001, *momentum*: 0.9
  4. **Loss:** *CrossEntropyLoss*, unweighed
  5. **Additional:** 
   - Number of samples per class was fixed to 4,552
   - Images are resized to 224x224 to meet ResNet Specifications
   - Model was retrained from scratch.
### **Observation**
 - The original hypothesis regarding Melanoma and Melanocytic Nevus being indistinguishable is confirmed
 - Although we didn't obtain an accuracy of 50% as expected, scores were still somewhat low
  <center>
    <div class="row" style="display: flex;">
    <div class="col" style="flex: 50%; padding: 5px;">
    <figure id="fig3"> 
    <img src="./images/acc_loss_melanocytic_exp2.1.png">
    <figcaption><p align="center">Fig.1 Accuracy Loss Curve for MEL vs NEV</p></figcaption>
    </figure>
    </div>
    <div class="col" style="flex: 50%; padding: 5px;">
    <figure id="fig3"> 
    <img src="./images/acc_loss_melanocytic_exp2.2.png">
    <figcaption><p align="center">Fig.2 Accuracy Comparision with Balanced Baseline</p></figcaption>
    </figure>
    </div>
    </div>
 </center>

 - [Figure 1](#fig3) The accuracy seems to always be above 50% meaning that we perform better than a random classifier. [Figure 2](#fig3) shows that this time the performance is always significantly above the 50% threshold apart from a single dip to 55%. However, this does not change the fact that a performance drop of almost 15% from [the previous experiment](#melanocytic-nevus-vs-melanoma) has been observed.
 #### *Classise Accuracy*

 |Class|Accuracy|
 |:----|---:|
 |1. Melanocytic Nevus|54.6%|
 |2. Melanoma |83.8%|

 |Mean Accuracy|Standard deviation|
 |:--:|:--:|
 |63.83%|4.69%|

 - The most interesting observation is the reversal of classwise accuracy scores for each class
 - This time, Melanoma performs far better (high recall (83.8% accuracy)) than Melanocytic Nevus (low recall (54.6% accuracy))
 - Classwise accuracy seems to be in favour of Melanoma as opposed to Melanocytic nevus when the samples are balanced.
 - Thus, Melanoma is easier to identify compared to Nevus, perhaps because it is present in an advanced stage in the dataset (see )

### **Conclusion**
 - Due to the switch in performance on the balanced dataset along with the low overall accuracy for a binary classification task, it can be concluded that Melanoma is not sufficiently distinguishable from melanocytic nevus. 
 - Roughly half the total Melanocytic Nevus samples are being misclassified.
 - On the other hand, the model seems good at identifying melanoma now. 
 
### **Further Analysis**
 *Note that this section contains plans for further analysis and not an extended analysis*
 
 - Comparing with the result of [the previous experiment](#melanocytic-nevus-vs-melanoma), it would seem that training against Melanoma after balancing the samples provides us with a good feature extractor for Melanoma.
 - This may have use in further experiments to retain the superclass label melanocytic across tasks.
 - Since these two are the largest occuring classes in the dataset, training this classification as the first task may serve a similar goal as EWC since due to lesser number of samples, the model will be unwilling to let go of previous task knowledge.
 - This can be confirmed by training with a reduced number of samples for the Melanoma vs Melanocytic Nevus case and seeing a negative trend for backward interference on the first task.
 - Another good idea would be seeing the effictiveness of this feature extractor for melanoma as compared to the previous obtained feature extractor for Nevus (presumably, since the performance on Nevus was high)
 - A line of questioning worth pursuing is that if these two classes are supposed to be visually indistinguishable for the most part, which samples are being classified consistently across runs? We know dysplastic nevi are similar to Melanoma, is it possible other samples such as blue Nevi may be present under the Melanocytic Nevus class? If this is the case, then are roughly 50% of the samples dysplastic Nevus since the classwise accuracy for nevus is only 54.6%? 
 - Explaining these result will provide significant insight into to exactly what the model has learnt about each class and will help us take a step towards the end goal of devising a curriculum for ordering tasks in continual learning, hopefully to achieve either better performance or more explainable models.

## **Melanoma vs Dysplastic Nevi**
# -------INCOMPLETE, REPORT RESULTS AND CONCLUSION-------------------
### **Aim**
 -  To ascertain the distinguishibility between melanoma and dysplastic nevus 
### **Procedure**
 - In this experiment we determine which samples in the Melanocytic nevus class were mislabelled as Melanoma by the classifier trained in [the previous experiment](#melanocytic-nevus-vs-melanoma-balanced).
 - We pass every misclassified nevus sample from `train_dataset` and `test_dataset` and store them in `train_dataset_refined` and `test_dataset_refined`
 - ~~The datasets are rebalanced to have an 90-10 ratio~~ Rebalancing was not required
 #### Dataset Stats before appending melanoma samples
 |Dataset| Number of Samples|
 |:---|---:|
 | `train_dataset_refined` | 4770 |
 | `test_dataset_refined` | 548 |

 #### Dataset Stats after appending melanoma samples
  |Dataset| Number of Samples|
  |:---|---:|
  | `train_dataset_refined` | 8840 |
  | `test_dataset_refined` | 1000 |

 #### *Experiment Details*
  1. **Architecture:** ResNet101
  2. **Train Epochs:** 14 
  3. **Optimizer:** *SGD*, *lr*: 0.001, *momentum*: 0.9
  4. **Additional:** 
   - Number of samples per class was fixed to 4,552
   - Images are resized to 224x224 to meet ResNet Specifications
   - Model was retrained from scratch.
### **Observation**
 - The original hypothesis regarding Melanoma and Melanocytic Nevus being indistinguishable is confirmed
 - Although we didn't obtain an accuracy of 50% as expected, scores were still somewhat low
  <center>
    <div class="row" style="display: flex;">
    <div class="col" style="flex: 50%; padding: 5px;">
    <figure id="fig3"> 
    <img src="./images/acc_loss_melanocytic_exp2.1.png">
    <figcaption><p align="center">Fig.1 Accuracy Loss Curve for MEL vs NEV</p></figcaption>
    </figure>
    </div>
    <div class="col" style="flex: 50%; padding: 5px;">
    <figure id="fig3"> 
    <img src="./images/acc_loss_melanocytic_exp2.2.png">
    <figcaption><p align="center">Fig.2 Accuracy Comparision with Balanced Baseline</p></figcaption>
    </figure>
    </div>
    </div>
 </center>

 - [Figure 1](#fig3) The accuracy seems to always be above 50% meaning that we perform better than a random classifier. [Figure 2](#fig3) shows that this time the performance is always significantly above the 50% threshold apart from a single dip to 55%. However, this does not change the fact that a performance drop of almost 15% from [the previous experiment](#melanocytic-nevus-vs-melanoma) has been observed.
 #### *Classise Accuracy*

 |Class|Accuracy|
 |:----|---:|
 |1. Melanocytic Nevus|94.2%|
 |2. Melanoma |24.1%|

 |Mean Accuracy|Standard deviation|
 |:--:|:--:|
 |61.78%|4.66%|

 - The most interesting observation is the reversal of classwise accuracy scores for each class
 - This time, Melanoma performs far better (high recall (83.8% accuracy)) than Melanocytic Nevus (low recall (54.6% accuracy))
 - Classwise accuracy seems to be in favour of Melanoma as opposed to Melanocytic nevus when the samples are balanced.
 - Thus, Melanoma is easier to identify compared to Nevus, perhaps because it is present in an advanced stage in the dataset (see )

### **Conclusion**