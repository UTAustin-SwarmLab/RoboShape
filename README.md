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
<img width="2700" height="1050" alt="furniture_presence_histogram" src="https://github.com/user-attachments/assets/efeba8fd-b521-4685-aee7-d2021b853d30" />
</div>

<div align="center">
<img width="2700" height="750" alt="inference_vs_gt_match_ratio" src="https://github.com/user-attachments/assets/f58be857-a279-4ef5-96ab-5d7c05b0e687" />
</div>

Distribution of numbers of points over all scenes as follows:
<div align="center">
<img width="1786" height="885" alt="point_count_histogram" src="https://github.com/user-attachments/assets/fefc0def-5249-4717-8467-33776621b189" />
</div>

Download & Setup:

Please request access to the ScanNet dataset and download it from the official ScanNet Benchmark: http://www.scan-net.org/.

