  <link rel="stylesheet" href="styles.css">

  <div class="scrollable-outline">
  <ul>
    <li><a href="#overview">Project Overview</a></li>
    <li><a href="#intro">Introduction</a></li>
    <li><a href="#methods">Methods</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#takeaways">Takeaways</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>  
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
<h2 id="overview" class="jump-link-target">Project Overview</h2>
Pulmonary edema is a condition in which there is an excess of fluid in the lungs. As pulmonary edema cases can quickly worsen and eventually lead to heart failure in cases of acute pulmonary edema, early and accurate detection is crucial in radiographs. Detecting pulmonary edema through radiographic assessment can be a challenge primarily because of the differences in each case. In this project, our main objective is to improve upon the model in the Deep Learning Radiographic Assessment of Pulmonary Edema paper, which had a goal of predicting the presence of pulmonary edema in radiographs using varying input sizes. In this project, we use transfer learning with an in-house lung image segmentation model from Hsiao Lab with the objective of isolating the lung region. We believe this reduces noise and focuses the network on the lungs, which is where edema would reside. We also plan to improve the existing ResNet152 model by adding labels for BMI and creatinine levels, which are relevant factors in presence of edema. These are relevant factors as creatinine levels can be used to measure how well your kidneys are working. With dramatically increased levels of creatinine, it can indicate potential renal failure, which can increase the presence of a pulmonary edema. (Farha Munguti, 2020). We will compare this model to the original model based on loss, Pearson R correlation between measured and predicted NT-pro B-type natriuretic peptide (BNPP) values, the non-active prohormone that is released in response to changes in pressure inside the heart, and the Area Under the Curve (AUC).

<h2 id="intro" class="jump-link-target">Introduction</h2>
Convolutional neural networks (CNNs) have been effective in classifying the presence of diseases in image data. In most cases, however, the data are labeled by physicians before being trained on, making this a supervised learning approach. Within the application of using chest radiographs to classify pulmonary edema, finding labeled data is often difficult due to how labor-intensive and subjective this process may be. Two radiologists may label mild cases of pulmonary edema differently since the indications of mild pulmonary edema would be Kerley B lines and peribronchial cuffing, which are more subtle. We can diagnose an individual with cardiogenic pulmonary edema if they have high B-type natriuretic peptide (BNP) and BNPP Based on scientific literature, the threshold for diagnosing cardiogenic pulmonary edema is BNPP > 400 or BNP > 100 (Kimand Januzzi Jr, 2011). A highly accurate neural network has the potential to save lives because cases can be diagnosed before symptoms worsen, allowing individuals with pulmonary edema to undergo the treatment they need. It will also be able to help individuals in Low and Middle-Income (LMIC) countries in which there may not be enough radiologists to interpret chest x-rays. In our replication of the model from the Deep Learning Radiographic Assessment of Pulmonary Edema paper by Justin Huynh et. al., we aimed to compare ResNet152 and VGG16 models to infer BNPP values from radiographs. However, with an existing lung image segmentation network, we are able to isolate each lung into two separate images, which we would then use our CNN models to train on. By having the original image, left lung, right lung, and combination of lungs with external data such as BMI, we hope this will allow us to better classify and quantify cardiogenic pulmonary edema. 

|   NTproBNP |   log10_NTproBNP |   bmi |   creatinine |   pneumonia |   acute_heart_failure |   cardiogenic_edema |
|-----------:|-----------------:|------:|-------------:|------------:|----------------------:|--------------------:|
|      418   |          2.62118 | 25.51 |         0.61 |           1 |                     0 |                   1 |
|     2161   |          3.33465 | 31.38 |         1.31 |           0 |                     0 |                   1 |
|      118   |          2.07188 | 33.81 |         0.66 |           0 |                     0 |                   0 |
|       49.9 |          1.6981  | 30.64 |         0.64 |           0 |                     0 |                   0 |
|    20029   |          4.30166 | 34.81 |        10.54 |           0 |                     0 |                   1 |

<iframe src="assets/BNPP_dist.html" width=800 height=600 frameBorder=0></iframe>
<iframe src="assets/logBNPP_dist.html" width=800 height=600 frameBorder=0></iframe>
<iframe src="assets/bmi_dist.html" width=800 height=600 frameBorder=0></iframe>
<iframe src="assets/creatinine_dist.html" width=800 height=600 frameBorder=0></iframe>


<h2 id="methods" class="jump-link-target">Methods</h2>
Add description about Methods

### Convolutional Neural Network (CNN) <a name="cnn_subparagraph"></a>
Prior to any network training, we preprocessed the data in order to keep the radiographs in a usable format for training purposes. We initially extracted the paths and keys for the radiographs with help from Jake in our section. Then, we used the existing training, validation, and test datasets and created a label for edema, which was 1 or 0, depending on the presence or absence of pulmonary edema. Since there were images that did not correspond to data in the existing datasets, we found the intersection and found that there were 15164 training, 1823 validation, and 1913 test set images. Around 80.2%, 9.6%, and 10.1% of the data corresponded to the training, validation, and test data, respectively. When preprocessing the data and creating the data loaders, we used a batch size of 32 and 4 workers.

The main model architectures we explored were VGG16 and ResNet152. Both architectures were trained on the L1-loss (mean absolute error) of the log BNPP values, as used in the original paper. We used a learning rate of 10e-4 and Adam optimizer while learning for 15 epochs for both architectures. When training the network, we unfroze layers and while optimizing the hyperparameters on the validation set, we froze layers. The ResNet152 model had 58145857 trainable parameters, while the VGG16 had 27514413. As in the paper, we used the ResNet152 model pre-trained on ImageNet. Since the VGG and ResNet architectures are generally used for classification, we altered the fully connected layer to have one output, which would be used for regression.

<center><img src="assets/Capstone Diagrams - Model1.png" alt="Model 1 Architecture" > </center>
<center><img src="assets/Capstone Diagrams - Model2.png" alt="Model 2 Architecture" ></center>
<center><img src="assets/Capstone Diagrams - Model3.png" alt="Model 3 Architecture" ></center>
<center><img src="assets/Capstone Diagrams - Model4.png" alt="Model 4 Architecture" ></center>

### Lung Segmentation <a name="segment_subparagraph"></a>
The lung segmentation network, given an input of a radiograph, would output six segmented images: right lung, left lung, heart, right clavicle, left clavicle, and spinal column. As seen in Figure TODO below, we can see six masks, one for each segment. The last image shows an overlaid graph of the segments, which was interesting to visualize, but not used for segmenting the images since it did not serve as a mask. Image masks should just be binary, such that multiplying the mask to an image will return the segment of interest from the original image.

Since edema is present in the lungs and a portion of the lungs are behind the heart, we decided to create a mask that combined the lung and heart segments. To create this, we used the numpy OR operator to include all the areas where the pixel value was 1. This yielded masks of the area of interest as seen in Figure TODO.

Finally, we applied these masks to the given images to produce the segmented images with area of interest. By simply multiplying the mask to the original image, we were able to produce an accurate isolated image of the lungs and heart. In Figure TODO, we can see examples of original images and their segmented counterparts, which would subsequently be fed into our neural network.
<center><img src="assets/Capstone Diagrams - Segmentation.png" alt="Segmentation in Action" ></center>



<h2 id="results" class="jump-link-target">Results</h2>
Add description about Findings

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



### Losses <a name="losses_subparagraph"></a>
Add description about losses - train and test

### AUROC Curves <a name="auroc_subparagraph"></a>
<iframe src="assets/ROC_Comparison.html" width=1000 height=600 frameBorder=0></iframe>

<h2 id="takeaways" class="jump-link-target">Takeaways</h2>
Add description about Takeaways - Conclusion

<h2 id="acknowledgments" class="jump-link-target">Acknowledgments</h2>
Add refs


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

