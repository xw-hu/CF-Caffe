# CF-Caffe
Caffe designed for Deep Context Features

## Basic Citation

If you use CF-Caffe, please cite:

@InProceedings{Hu_2018_CVPR,      
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Zhu, Lei and Fu, Chi-Wing and Qin, Jing and Heng, Pheng-Ann},      
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-Aware Spatial Context Features for Shadow Detection},      
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},      
&nbsp;&nbsp;&nbsp;&nbsp;  pages={7454--7462},        
&nbsp;&nbsp;&nbsp;&nbsp;  year = {2018}      
}

@article{hu2019direction,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Fu, Chi-Wing and Zhu, Lei and Qin, Jing and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-Aware Spatial Context Features for Shadow Detection and Removal},    
&nbsp;&nbsp;&nbsp;&nbsp;  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2019},          
&nbsp;&nbsp;&nbsp;&nbsp;  note={to appear}                  
}

@article{jia2014caffe,       
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},       
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Caffe: Convolutional Architecture for Fast Feature Embedding},       
&nbsp;&nbsp;&nbsp;&nbsp;  journal = {arXiv preprint arXiv:1408.5093},       
&nbsp;&nbsp;&nbsp;&nbsp;  year = {2014}       
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


## Models

If you use these models, please cite their papers accordingly.


1. Segmentation models in `examples/segmentation/`:

   Deeplab v1, Deeplab v3, Deeplab v3 plus, PSPNet, PSANet, Non-local Network (FPN based).
   
   
2. This version of Caffe is used in:

### [DAF-Net](https://github.com/xw-hu/DAF-Net)
@InProceedings{Hu_2019_CVPR,      
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Fu, Chi-Wing and Zhu, Lei and Heng, Pheng-Ann},      
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Depth-Attentional Features for Single-Image Rain Removal},      
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},      
&nbsp;&nbsp;&nbsp;&nbsp;  pages={8022--8031},      
&nbsp;&nbsp;&nbsp;&nbsp;  year = {2019}      
}

### [DSC](https://github.com/xw-hu/DSC)
@InProceedings{Hu_2018_CVPR,      
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Zhu, Lei and Fu, Chi-Wing and Qin, Jing and Heng, Pheng-Ann},      
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-Aware Spatial Context Features for Shadow Detection},      
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},      
&nbsp;&nbsp;&nbsp;&nbsp;  pages={7454--7462},        
&nbsp;&nbsp;&nbsp;&nbsp;  year = {2018}      
}

@article{hu2019direction,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Fu, Chi-Wing and Zhu, Lei and Qin, Jing and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Direction-Aware Spatial Context Features for Shadow Detection and Removal},    
&nbsp;&nbsp;&nbsp;&nbsp;  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2019},          
&nbsp;&nbsp;&nbsp;&nbsp;  note={to appear}                  
}

### [GNLB](https://github.com/xw-hu/GNLB)
@article{zhu2018saliency,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Zhu, Lei and Hu, Xiaowei and Fu, Chi-Wing and Qin, Jing and Heng, Pheng-Ann},    
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Saliency-Aware Texture Smoothing},    
&nbsp;&nbsp;&nbsp;&nbsp;  journal={IEEE Transactions on Visualization and Computer Graphics},    
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2018}    
}

### [RADF](https://github.com/xw-hu/RADF)
@inproceedings{hu18recurrently,   
&nbsp;&nbsp;&nbsp;&nbsp;  author = {Hu, Xiaowei and Zhu, Lei and Qin, Jing and Fu, Chi-Wing and Heng, Pheng-Ann},              
&nbsp;&nbsp;&nbsp;&nbsp;  title = {Recurrently Aggregating Deep Features for Salient Object Detection},    
&nbsp;&nbsp;&nbsp;&nbsp;  booktitle = {AAAI},   
&nbsp;&nbsp;&nbsp;&nbsp;  pages={6943--6950},         
&nbsp;&nbsp;&nbsp;&nbsp;  year  = {2018}    
}
