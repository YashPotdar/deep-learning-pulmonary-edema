<script type="text/x-mathjax-config"> MathJax.Hub.Config({ TeX: { equationNumbers: { autoNumber: "all" } } }); </script>
   <script type="text/x-mathjax-config">
     MathJax.Hub.Config({
       tex2jax: {
         inlineMath: [ ['$','$'], ["\\(","\\)"] ],
         processEscapes: true
       }
     });
   </script>
   <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script> 
 
 <link rel="stylesheet" href="styles.css">

  <div class="scrollable-outline">
  <ul>
    <li><a href="#motivation">Motivation</a></li>
    <li><a href="#overview">Project Overview</a></li>
    <li><a href="#data">Data</a></li>
    <li><a href="#methods">Methods</a></li>
    <li><a href="#findings">Findings</a></li>
    <li><a href="#takeaways">Takeaways</a></li>
    <li><a href="#conclusion">Conclusion</a></li>  
    <li><a href="#about_us">About Us</a></li>
  </ul>
</div>

<style>
  .scrollable-outline {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 50;
    overflow-x: scroll;
    white-space: nowrap;
    background-color: #F0F0F0;
  }

  .scrollable-outline ul {
    display: inline-block;
    margin: 0;
    padding: 100;
  }

  .scrollable-outline li {
    display: inline-block;
    margin-right: 20px;
    font-size: 18px;
  }
</style>


## David Davila-Garcia, Yash Potdar, Marco Morocho

### Advisor: Dr. Albert Hsiao 

<img src="assets/UCSanDiego_Halicioglu_BlueGold.png" alt="HDSI logo" height="100">

 <div class="button-container">  
  <details class="button">
    <summary>View Poster</summary>
    <br>
    <img src="assets/DeepLearningEdemaPoster.png" alt="Poster Presentation" >
</details>
  
  <a href="https://github.com/ddav118/DSC-180B"  target="_blank" class="button">Project Repository</a>
  </div>

---
---

<h2 id="motivation" class="jump-link-target">Motivation</h2>
Pulmonary edema is a serious and potentially life-threatening condition caused by increased extravascular lung water. Cardiogenic pulmonary edema (CPE) results from increased blood pressure and is a result of heart failure.

Current methods of diagnosis for CPE depend on radiologists manually examining X-ray images. Manual examination is time-consuming and may not be a luxury that patients can afford to wait on. Moreover, manual work done by radiologists is not 100% accurate, leaving room for human error. A highly accurate neural network has the potential to help individuals with CPE receive necessary treatment in a timely manner, particularly in low and middle-income countries where there may not be enough radiologists to interpret chest X-rays.

<h2 id="overview" class="jump-link-target">Project Overview</h2>

Convolutional neural networks (CNNs) have been effective in classifying diseases from medical images. In this project, we aim to build upon the methods used by Justin Huynh in “Deep Learning Radiographic Assessment of Pulmonary Edema”. This paper demonstrated the potential of CNNs in diagnosing CPE by training them on chest radiographs and NT-proBNP, a clinical biomarker measured from blood serum. In this study, we examine the effects of the addition of clinical data and image segmentation. We believe the addition of the following can improve the accuracy of a CNN classifier of CPE:

- **Clinical data**: Recent literature suggests that NT-proBNP concentrations can be influenced by confounding factors such as renal failure, age, sex, and body mass index (BMI). Thus, we will train the model with confounding clinical data of BMI, creatinine level, and presence of pneumonia and acute heart failure to correctly distinguish normal and edema cases.

- **Lung and Heart Segmentation**: We will use transfer learning with an image segmentation model to isolate the heart and lungs of an X-ray image. We believe this reduces noise and focuses the network on the regions where edema would reside.

<h2 id="data" class="jump-link-target">Data</h2>
We constructed a dataset of 16,619 records from UC San Diego (UCSD) Health patients. The dataset initially provided by the UCSD Artificial Intelligence and Data Analytics (AIDA) Lab had 18,900 records, but we dropped rows with missing clinical data. We also removed the unique identifiers for patients.

|   NTproBNP |   log10_NTproBNP |   bmi |   creatinine |   pneumonia |   acute_heart_failure |   cardiogenic_edema |
|-----------:|-----------------:|------:|-------------:|------------:|----------------------:|--------------------:|
|      418   |          2.62118 | 25.51 |         0.61 |           1 |                     0 |                   1 |
|     2161   |          3.33465 | 31.38 |         1.31 |           0 |                     0 |                   1 |
|      118   |          2.07188 | 33.81 |         0.66 |           0 |                     0 |                   0 |
|       49.9 |          1.6981  | 30.64 |         0.64 |           0 |                     0 |                   0 |
|    20029   |          4.30166 | 34.81 |        10.54 |           0 |                     0 |                   1 |

The `NT-proBNP` column represents the NT-proBNP value, a continuously valued biomarker measured from blood serum samples. As seen in the distribution below, there is a strong right skew due to the abnormally high NT-proBNP values. We performed a log transformation to create the `log10_NTproBNP` column. Using the threshold for pulmonary edema established in Huynh’s paper and prior work, we classified any patient with an NT-proBNP value of at least $400 pg/mL$ as an edema case. Any records with a log NT-proBNP value of at least $2.602$ ($log_{10} 400$) are considered edema cases. 

The `bmi` column contained the body mass index ($kg/m^2$) of the patient, which is derived from a patient’s mass and height. The ‘creatinine’ column contains a continuous value of creatinine level ($mg/dL$) measured from blood serum samples. The `pneumonia` and `acute_heart_failure` columns contain binary values and are 1 if a patient has the condition. In the dataset, 12.0% of patients have pneumonia and 17.2% have acute heart failure. The distributions of the quantitative features are shown below.

The target column (`cardiogenic_edema`) contains binary values which are 1 if a patient has CPE. Around 64.7% of the records in our dataset had CPE based on the threshold. 

<iframe src="assets/BNPP_dist.html" width=900 height=630 frameBorder=0></iframe>
<iframe src="assets/logBNPP_dist.html" width=900 height=630 frameBorder=0></iframe>
<iframe src="assets/bmi_dist.html" width=900 height=630 frameBorder=0></iframe>
<iframe src="assets/creatinine_dist.html" width=900 height=630 frameBorder=0></iframe>


<h2 id="methods" class="jump-link-target">Methods</h2>
We trained four modified ResNet152 CNN architectures with differing inputs: (A) Original Radiographs only, (B) Original Radiographs + Clinical Data, (C) Original Radiographs + Heart & Lung Segmentations, and (D) Original Radiographs + Heart & Lung Segmentations + Clinical Data. The data were randomly split into train, validation, and test sets at a ratio of 80%/10%/10%. The four model’s accuracy and AUC on the test set (n = 1,662) were used to compare model performance.

### Input: Clinical Data <a name="clinical_subparagraph"></a>
To ensure high-quality data for our project, we excluded patients with missing values for columns containing clinical data, specifically BMI, creatinine, pneumonia, and acute heart failure. These values which have been identified as confounds for CPE would be appended to the feature vector within the ResNet152 CNN.

### Input: Lung & Heart Image Segmentation <a name="segmentation_subparagraph"></a>
The UCSD AIDA laboratory provided us with a pre-trained U-Net CNN, which creates predicted binary masks of the right and left lungs, heart, right and left clavicle, and spinal column for each patient's X-ray. A left lung mask, as seen in the diagram below, has a value of 1 for pixels representing part of the left lung, and 0 otherwise. Masks can be combined using an `OR` operation, which yields a value of 1 when either mask has a value of 1, and 0 otherwise. In order to segment an image, we would simply multiply the binary mask to the image, which would yield an image with pixels corresponding to the 1’s in a mask.

In our segmentation process, we applied the pre-trained model to the full dataset of X-rays. We then used the binary masks to create two segmented images of the lungs and heart. The segmentation process is visualized in the following figure, where we begin with the original radiograph, generate masks of the lungs and heart regions, and apply it over the radiograph to isolate the lungs and heart.
<center><img src="assets/Capstone Diagrams - Segmentation.png" alt="Segmentation in Action" ></center>

### Model Architectures <a name="architectures_subparagraph"></a>
We used the default PyTorch ResNet152 model with a regression output since we were predicting `log10_NTproBNP` values. By using the classification threshold for `log10_NTproBNP` of 2.602, we were able to make a classification output. The four architectures, which differ by their inputs, are shown below:
- **Model A**: original X-rays
<center><img src="assets/Capstone Diagrams - Model1_log_output.png" alt="Model 1 Architecture" > </center>
- **Model B**: original X-rays + clinical data
   - Concatenates the clinical data with the output from the convolutional layers
<center><img src="assets/Capstone Diagrams - Model2_log_output.png" alt="Model 2 Architecture" ></center>
- **Model C**: original X-rays + lung segmentations + heart segmentations
<center><img src="assets/Capstone Diagrams - Model3_log_output.png" alt="Model 3 Architecture" ></center>
- **Model D**: original X-rays + lung segmentations + heart segmentations + clinical data
   - Concatenates the clinical data with the output from the convolutional layers
<center><img src="assets/Capstone Diagrams - Model4_log_output.png" alt="Model 4 Architecture" ></center>

### Model Training & Testing <a name="train_test_subparagraph"></a>
All four models were trained on the training set for 20 epochs using the Nadam optimizer with a learning rate of 0.001, and the mean absolute error (MAE) was used as the loss function to evaluate the neural network. The MAE on the validation set was computed after each epoch, and early stopping was implemented such that the model with the minimum MAE on the validation set was saved. After 20 epochs, the MAE on the unseen test set was computed for the four models with the minimum MAE on the validation set. We saved the predicted `log10_NTproBNP` values for each patient in the test set and compared these to the laboratory-measured values to evaluate our models. 

<h2 id="findings" class="jump-link-target">Findings</h2>
In order to evaluate our models, we compared how the predicted `log10_NTproBNP` compared to the laboratory-measured `log10_NTproBNP` values. We did this by plotting predictions against true values and calculating the Pearson correlation coefficient. We also used the threshold for CPE to binarize each model's predicted values of `log10_NTproBNP` and calculate accuracy and Area under the ROC Curve (AUC).

The table below exhibits the ResNet152 model performances by input data, including their respective Train L1-Loss, Test L1-Loss, Accuracy, AUC, and Pearson R scores, highlighting that **Model (B) performed the best** with an accuracy of 0.787 and AUC of 0.869. It is noteworthy that Model (D) performed marginally worse than Model (B). Therefore, the results suggest that incorporating clinical data positively impacted the ResNet152 model's performance to identify cases of CPE, but the addition of heart and lung segmentation did not.

<center> <img src="assets/Capstone Diagrams - Results.png" alt="Test Set Results" height="400"></center>


### Confusion Matrices <a name="confusion_matrices"></a>

<iframe src="assets/ModelA_Confusion_Matrix.html" width=700 height=600 frameBorder=0></iframe>
<iframe src="assets/ModelB_Confusion_Matrix.html" width=700 height=600 frameBorder=0></iframe>
<iframe src="assets/ModelC_Confusion_Matrix.html" width=700 height=600 frameBorder=0></iframe>
<iframe src="assets/ModelD_Confusion_Matrix.html" width=700 height=600 frameBorder=0></iframe>

### Pearson R Correlation Comparison <a name="correlation"></a>

<iframe src="assets/ModelA_Correlation.html" width=700 height=600 frameBorder=0></iframe>
<iframe src="assets/ModelB_Correlation.html" width=700 height=600 frameBorder=0></iframe>
<iframe src="assets/ModelC_Correlation.html" width=700 height=600 frameBorder=0></iframe>
<iframe src="assets/ModelD_Correlation.html" width=700 height=600 frameBorder=0></iframe>

### AUROC Curves <a name="auroc_subparagraph"></a>
<iframe src="assets/ROC_Comparison.html" width=1100 height=800 frameBorder=0></iframe>

<h2 id="takeaways" class="jump-link-target">Takeaways</h2>
From our results above, it is apparent that a more complex and preprocessed input did not improve the ability of the classifier to identify cases of pulmonary edema. We had hypothesized that image segmentation of X-rays would help focus the neural network on the lungs and heart regions, which is where edema occurs. We believed by reducing noise in the input, the classifier would be able to better distinguish normal and edema cases. However, segmentation did not improve performance. We propose two possible reasons for these findings:

- The neural network already focuses on the relevant regions of the X-rays where edema occurs. Due to this, providing segmented images is not necessarily adding information to the ResNet152 model, but is instead cropping information out when it is passed to the network. 

- The dataset size may have been sufficient for the neural network to distinguish normal and edema cases without the need for segmentation.  With around 13,000 training images, there was enough data for the neural network to learn the features that distinguish normal and edema cases in the lung and heart regions. A smaller dataset may benefit from segmentation as it provides more focused input and could help the neural network to learn the relevant features more accurately. Segmentation may yield a better classifier, in this case, due to the lack of the neural network’s ability to have enough data to focus on the heart and lungs.

Overall, while we had expected segmentation to improve the classifier's accuracy, our results suggest that it may not always be necessary or beneficial depending on the dataset size and the neural network's ability to learn relevant features.


<h2 id="conclusion" class="jump-link-target">Conclusion</h2>
This project demonstrates the relevance of considering confounding factors, clinical data, and image segmentation when training CNN models to diagnose CPE from chest radiographs. This project potentially has a strong impact because a highly accurate neural network can diagnose cases of CPE before symptoms worsen, allowing individuals to undergo early treatment. This could increase equity and allow individuals with reduced access to healthcare receive timely diagnoses. While an increase in CNN model performance was observed from adding clinical data, no such change was observed from heart and lung segmentation. Further research is needed to determine the optimal use of image segmentation in CNN models.


<h2 id="acknowledgments" class="jump-link-target">Acknowledgments</h2>
We would like to thank our mentor, Dr. Albert Hsiao, for his guidance and feedback throughout the duration of this project. We would also like to thank Amin Mahmoodi in the UCSD AIDA lab for sharing the lung and heart segmentation network with us. Finally, we would like to thank our fellow students in our capstone section who we were able to learn from and collaborate with throughout the last six months.


---
---

<h2 id="about_us" class="jump-link-target">About Us</h2>


<table style="margin:auto; border-collapse: collapse;">
  <tr>
    <td style="border-right: 1px solid black;"><a href="https://www.linkedin.com/in/david-d%C3%A1vila-garc%C3%ADa-001/" target="_blank">David Davila-Garcia</a></td>
    <td style="border-right: 1px solid black;"><a href="https://www.linkedin.com/in/yashmpotdar/" target="_blank">Yash Potdar</a></td>
    <td><a href="https://www.linkedin.com/in/marco-morocho-0062641b9/" target="_blank">Marco Morocho</a></td>
  </tr>
  <tr>
    <td style="border-right: 1px solid black;">Pic 1</td>
    <td style="border-right: 1px solid black;">Pic 2</td>
    <td>Pic 3</td>
  </tr>
</table>
