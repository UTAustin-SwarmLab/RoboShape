# RoboShape
Information Theoretical Pipeline for Privacy Preserving Intelligent Robotics Sensing

# Feature Extractor: Sonata
We used PTv3 pre-trained model Sonata as feature extractor in order to use point cloud modalities.
You can find the model details here: https://github.com/facebookresearch/sonata

At the end of the feature extractor encoder layers, model supplies points with 512 dimensions. 

# Dataset: Scannet
This project utilizes the Scannet Dataset for 3D object detection. 

Visualizations:

Below are samples of the point clouds visualizations obtained by running sonata_ınference.py using Scannet Scenes:

<table width="100%">
  <tr>
    <td align="center" width="20%">
      <img src="https://github.com/user-attachments/assets/a893dfcc-2025-4960-b2ca-144c7d5328d6" width="100%">
      <br>
      <i>Figure 1</i>
    </td>
    <td align="center" width="20%">
      <img src="https://github.com/user-attachments/assets/6158422e-14df-4587-b4e1-2717d3293819" width="100%">
      <br>
      <i>Figure 2</i>
    </td>
    <td align="center" width="20%">
      <img src="https://github.com/user-attachments/assets/0ea08317-72a0-4a07-90b9-d1e4342fdc60" width="100%">
      <br>
      <i>Figure 3</i>
    </td>
    <td align="center" width="20%">
      <img src="https://github.com/user-attachments/assets/ba744e2a-8242-4010-bb42-1a468e2fc4cf" width="100%">
      <br>
      <i>Figure 4</i>
    </td>
    <td align="center" width="20%">
      <img src="https://github.com/user-attachments/assets/44f7ffd1-a23a-4a1b-b55e-fe407375af32" width="100%">
      <br>
      <i>Figure 5</i>
    </td>
  </tr>
</table>

The furniture distributions over scenes , and the comparison of ground-truth lables and the segmentation of the Sonata are as follows.

<div align="center">
<img width="2700" height="1050" alt="furniture_presence_histogram" src="https://github.com/user-attachments/assets/ed0f2cb9-54e9-4635-a398-95aedfea5b26" />

</div>

<div align="center">
<img width="2700" height="750" alt="inference_vs_gt_match_ratio" src="https://github.com/user-attachments/assets/719ecb55-631e-4beb-b948-702fdb60548d" />

</div>

Distribution of numbers of points over all scenes as follows:

<div align ="center">
<img width="1666" height="1038" alt="point_count_histogram" src="https://github.com/user-attachments/assets/5d3e385d-8d32-491c-9bb9-16ceb51868db" />

</div>
<div align ="center">
<img width="2233" height="447" alt="scene_type_colormap" src="https://github.com/user-attachments/assets/2ecf4234-890e-4ab3-a794-dc66798220a6" />
</div>

Distributions of furnitures over scene types :
<div align= "center">
<img width="3578" height="1777" alt="scene_type_furniture_distribution" src="https://github.com/user-attachments/assets/88d72703-0852-4d96-b030-11296c92f7bc" />
</div>


Sonata Encoder unites different points during downsampling the raw points in the encoding process, find the number of different furniture lables for each voxel for scannet dataset below:

<div align= "center">
<img width="1786" height="884" alt="voxel_label_diversity" src="https://github.com/user-attachments/assets/97fe11ba-ae70-4974-8abb-7ceece63c951" />

</div>

Download & Setup:

Please request access to the ScanNet dataset and download it from the official ScanNet Benchmark: http://www.scan-net.org/.
# Training:

# Scannet
<div align= "center">
<img width="2117" height="865" alt="mi_curves_epoch0_19" src="https://github.com/user-attachments/assets/52dd6d94-ab4f-4864-8b62-67eeaf746a21" />
</div>

# Matterport3D

<div align= "center">
<img width="1350" height="900" alt="roboshape_train_val_loss_matterport" src="https://github.com/user-attachments/assets/f036ed4c-6c7c-4d2b-aab9-b13ebdb0fd25" />

</div>
<div align= "center">
  <img width="2117" height="865" alt="RoboShape_training_curves_matterport3D" src="https://github.com/user-attachments/assets/81d76029-cb8c-45e1-8d0d-34be92681d09" />


</div>

# ARKitScenes

<div align= "center">
<img width="1350" height="900" alt="train_val_loss_arkitscenes" src="https://github.com/user-attachments/assets/8482bbaf-2de8-4d26-9ad9-20d765c62da0" />

</div>
<div align= "center">
<img width="2117" height="865" alt="training_curves_arkitscenes_subset2" src="https://github.com/user-attachments/assets/10f8b029-0f8c-4799-864a-a0333a224cbf" />



</div>



# Results: 
# Scannet 
4 different classifiers trained in order to show the success of roboshape embeddings at hiding private attributes. 2 of them trained in order to classify sonata embeddings according to furniture type (public label) and scenetype ( private label), and the other 2 of them in order to classify roboshape embeddings according to public and private labels. You can find the Train , test losses and classifying accuracies below.
<div align= "center">
<img width="1271" height="359" alt="public_classsifiers_loss_scannet" src="https://github.com/user-attachments/assets/b3281bfa-3fd2-4f51-9f1a-a8f268dbb3e2" />
</div>

<div align= "center">

<img width="1275" height="366" alt="private_classifiers_loss_scannet" src="https://github.com/user-attachments/assets/c4fcefba-f310-4591-b749-b544caaf4095" />

</div>

Loss curves of classifiers for noisy original embeddings:
<div align= "center">
<img width="1800" height="600" alt="noisy_classifier_loss_scannet" src="https://github.com/user-attachments/assets/38de6cf0-c7cc-4915-bec2-3213fa95be79" />


</div>

Loss curves of classifiers for randomly initialized encoder outputs:

<div align= "center">
<img width="1800" height="600" alt="random_encoder_classifier_loss_scannet" src="https://github.com/user-attachments/assets/9354ce5d-26fd-4f7a-adca-d2b6a7dc6a21" />
</div>


Auroc results for 4 different baselines. 
<div align= "center">
  <img width="1272" height="532" alt="auroc_scannet" src="https://github.com/user-attachments/assets/d2adcc0f-c2f9-430b-b107-ad7260e30974" />
</div>

# Matterport
4 different classifiers trained in order to show the success of roboshape embeddings at hiding private attributes. 2 of them trained in order to classify sonata embeddings according to furniture type (public label) and scenetype ( private label), and the other 2 of them in order to classify roboshape embeddings according to public and private labels. You can find the Train , test losses and classifying accuracies below.
<div align= "center">
<img width="2100" height="600" alt="public_classifier_loss_matterport3D" src="https://github.com/user-attachments/assets/b3137b0e-47b5-4733-84be-c1d789ecc0c3" />

</div>

<div align= "center">

<img width="2100" height="600" alt="private_classifier_loss_matterport3D" src="https://github.com/user-attachments/assets/03979693-1fb6-473f-ace3-3a64570e629e" />

</div>

Loss curves of classifiers for noisy original embeddings:
<div align= "center">
<img width="1800" height="600" alt="noisy_classifer_loss_matterport3D" src="https://github.com/user-attachments/assets/a225eb25-c3f0-4802-899b-8601bdc1dd4e" />


</div>

Loss curves of classifiers for randomly initialized encoder outputs:

<div align= "center">
<img width="1800" height="600" alt="random_encoder_loss_matterport3D" src="https://github.com/user-attachments/assets/5397eb19-caaa-434a-9e29-a2547a3ca086" />

</div>
Auroc results for 4 different baselines:
<div align= "center">
<img width="1800" height="750" alt="auroc_matterport3D" src="https://github.com/user-attachments/assets/902d6f92-0d66-4732-8a04-f3cc1359f437" />


</div>

# ARKitScenes

4 different classifiers trained in order to show the success of roboshape embeddings at hiding private attributes. 2 of them trained in order to classify sonata embeddings according to furniture type (public label) and scenetype ( private label), and the other 2 of them in order to classify roboshape embeddings according to public and private labels. You can find the Train , test losses and classifying accuracies below.
<div align= "center">
<img width="2100" height="600" alt="loss_public_arkitscenes_subset2" src="https://github.com/user-attachments/assets/9d55e55a-9e9e-46ba-9a42-e24512d59cfc" />

</div>

<div align= "center">

<img width="2100" height="600" alt="loss_private_arkitscenes_subset2" src="https://github.com/user-attachments/assets/7a815312-50c1-41c6-827e-d76d44ef4b89" />


</div>

Loss curves of classifiers for noisy original embeddings:
<div align= "center">
<img width="1800" height="600" alt="loss_noisy_arkitscenes_subset2" src="https://github.com/user-attachments/assets/fee000c9-20ae-4766-8a2a-1032c07bc59b" />



</div>

Loss curves of classifiers for randomly initialized encoder outputs:

<div align= "center">
<img width="1800" height="600" alt="loss_random_encoder_arkitscenes_subset2" src="https://github.com/user-attachments/assets/cb55d23d-6edb-4ceb-aceb-a16cad0d793a" />

</div>


Auroc results for 4 different baselines. 
<div align= "center">
<img width="1800" height="750" alt="auroc_curves_arkitscenes_subset2" src="https://github.com/user-attachments/assets/b210a212-0f60-4a3f-aff9-33aaa2580754" />

</div>
