# CF-Caffe
Caffe designed for Deep Context Features

## Basic Citation

If you use CF-Caffe, please cite:

@InProceedings{Hu_2018_CVPR,      
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Zhu, Lei and Fu, Chi-Wing and Qin, Jing and Heng, Pheng-Ann},      
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-Aware Spatial Context Features for Shadow Detection},      
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},      
&nbsp;&nbsp;&nbsp;&nbsp;  month = {June},      
&nbsp;&nbsp;&nbsp;&nbsp;  year = {2018}      
}

@article{hu2018direction,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Fu, Chi-Wing and Zhu, Lei and Qin, Jing and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-aware Spatial Context Features for Shadow Detection and Removal},    
&nbsp;&nbsp;&nbsp;&nbsp;  journal={arXiv preprint arXiv:1805.04635},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2018}    
}

@article{jia2014caffe,       
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},       
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Caffe: Convolutional Architecture for Fast Feature Embedding},       
&nbsp;&nbsp;&nbsp;&nbsp;  journal = {arXiv preprint arXiv:1408.5093},       
&nbsp;&nbsp;&nbsp;&nbsp;  year = {2014}       
}

## Models

1. Basic models in `examples/segmentation/`:

   Deeplab v1, Deeplab v3, Deeplab v3 plus, PSPNet, PSANet, Non-local Network (FPN based).
   
   If you use these models, please cite their papers accordingly.
   
2. This version of Caffe is used in:


### DSC
@InProceedings{Hu_2018_CVPR,      
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Zhu, Lei and Fu, Chi-Wing and Qin, Jing and Heng, Pheng-Ann},      
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-Aware Spatial Context Features for Shadow Detection},      
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},      
&nbsp;&nbsp;&nbsp;&nbsp;  month = {June},      
&nbsp;&nbsp;&nbsp;&nbsp;  year = {2018}      
}

@article{hu2018direction,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Fu, Chi-Wing and Zhu, Lei and Qin, Jing and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-aware Spatial Context Features for Shadow Detection and Removal},    
&nbsp;&nbsp;&nbsp;&nbsp;  journal={arXiv preprint arXiv:1805.04635},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2018}    
}

### GNLB
To appear.

### RADF
@inproceedings{hu18recurrently,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Zhu, Lei and Qin, Jing and Fu, Chi-Wing and Heng, Pheng-Ann},              
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Recurrently Aggregating Deep Features for Salient Object Detection},    
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {AAAI},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2018}    
}

## Installation
1. Clone this repository.

    ```shell
    git clone https://github.com/xw-hu/CF-Caffe.git
    ```

2. Build CF-Caffe

   *This model is tested on Ubuntu 16.04, CUDA 8.0.
    
   Follow the Caffe installation instructions here: [http://caffe.berkeleyvision.org/installation.html](http://caffe.berkeleyvision.org/installation.html)   
   
   ```shell
   make all -jXX
   ```
   
3. If you want to use MATLAB or Python:

   ```shell
   make matcaffe
   make pycaffe
   ```
