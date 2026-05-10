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
<div align= "center">
<img width="2117" height="865" alt="mi_curves_epoch0_19" src="https://github.com/user-attachments/assets/52dd6d94-ab4f-4864-8b62-67eeaf746a21" />
</div>

# Results: 
4 different classifiers trained in order to show the success of roboshape embeddings at hiding private attributes. 2 of them trained in order to classify sonata embeddings according to furniture type (public label) and scenetype ( private label), and the other 2 of them in order to classify roboshape embeddings according to public and private labels. You can find the Train , test losses and classifying accuracies below.
<div align= "center">
<img width="1800" height="600" alt="loss_public" src="https://github.com/user-attachments/assets/f751736c-53b8-4292-9711-1f7ab6c4f896" />

</div>
<div align= "center">
<img width="1800" height="600" alt="loss_private" src="https://github.com/user-attachments/assets/9f057cca-07b4-47df-afd6-8b03cd31880c" />
</div>

Loss curves of classifiers for noisy original embeddings:
<div align= "center">
<img width="1800" height="600" alt="loss_noisy" src="https://github.com/user-attachments/assets/9ba05ddb-1861-4084-b3ce-45f9f0a97948" />

</div>

Loss curves of classifiers for randomly initialized encoder outputs:

<div align= "center">
<img width="1800" height="600" alt="loss_random_encoder" src="https://github.com/user-attachments/assets/43f6d727-0b18-48fc-b06c-d3984457b6e1" />


</div>
