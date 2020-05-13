// ------------------------------------------------------------------
// Copyright (c) 2018
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
//
// recurrently-attenuating translation
// y(i,j) = alpha*f(i-1,j)+f(i,j)
// if y(i,j)>=0, y(i,j)=y(i,j) else y(i,j)=beta*y(i,j)
// ------------------------------------------------------------------

#include <vector>

#include "caffe/layers/rat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>  //left up right down
__global__ void RATForward(const int nthreads, const Dtype* bottom_data, Dtype* top_left, Dtype* top_up, Dtype* top_right, Dtype* top_down, const int channel, const int height, const int width, const Dtype* weight_data, const Dtype* slope_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {

     int w = index % width;
     int h = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     Dtype temp = 0; 

     //left
     top_left[index] = 0; 
     
     for (int i=width-1; i>=w; i--)
     {
        temp = top_left[index] * weight_data[c] + bottom_data[((n*channel + c)*height + h)*width + i]; //0*channel+c
        top_left[index] = (temp > 0) ? temp : slope_data[c]*temp; // x=x if x>0, else x=slope*x 
     }

      
     //up
     top_up[index] = 0;
 
     for (int i=height-1; i>=h; i--)
     {
        temp = top_up[index] * weight_data[channel + c] + bottom_data[((n*channel + c)*height + i)*width + w]; //1*channel+c
        top_up[index] = (temp > 0) ? temp : slope_data[c]*temp; // x=x if x>0, else x=slope*x 
     }
     
     
     //right
     top_right[index] = 0;
     
     for (int i=0; i<=w; i++)
     {
        temp = top_right[index] * weight_data[2*channel + c] + bottom_data[((n*channel + c)*height + h)*width + i]; //2*channel+c
        top_right[index] = (temp > 0) ? temp : slope_data[c]*temp; // x=x if x>0, else x=slope*x 
     }
     
     
     //down
     top_down[index] = 0; 

     for (int i=0; i<=h; i++)
     {
        temp = top_down[index] * weight_data[3*channel + c] + bottom_data[((n*channel + c)*height + i)*width + w]; //3*channel+c
        top_down[index] = (temp > 0) ? temp : slope_data[c]*temp; // x=x if x>0, else x=slope*x 
    
     }

     //printf("ori: %f, left: %f, up: %f, right: %f, down: %f, n: %d, c: %d, h:%d, w:%d\n",bottom_data[index], top_left[index], top_up[index], top_right[index], top_down[index],n,c,h,w);
  }
}

template <typename Dtype>
__global__ void RATBackward(const int nthreads, Dtype* bottom_diff, const Dtype* top_left, const Dtype* top_up, const Dtype* top_right, const Dtype* top_down, const Dtype* top_left_data, const Dtype* top_up_data, const Dtype* top_right_data, const Dtype* top_down_data, const int channel, const int height, const int width, const Dtype* weight_data, const Dtype* slope_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
     
     int w = index % width;
     int h = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     Dtype diff_right = 0;
     Dtype diff_left = 0;
     Dtype diff_down = 0;
     Dtype diff_up = 0;


     //right 
     for (int i=width-1; i>=w; i--)
     {
        diff_right *= weight_data[2*channel + c];
        diff_right += top_right[((n*channel + c)*height + h)*width + i];
        diff_right *= (top_right_data[((n*channel + c)*height + h)*width + i]<=0)? slope_data[c] : 1;
     }
 

     //left 
     for (int i=0; i<=w; i++)
     {  
        diff_left *= weight_data[c];
        diff_left += top_left[((n*channel + c)*height + h)*width + i];
        diff_left *= (top_left_data[((n*channel + c)*height + h)*width + i]<=0)? slope_data[c] : 1;
     }
     

     //down 
     for (int i=height-1; i>=h; i--)
     {
        diff_down *= weight_data[3*channel + c];
        diff_down += top_down[((n*channel + c)*height + i)*width + w];
        diff_down *= (top_down_data[((n*channel + c)*height + i)*width + w]<=0)? slope_data[c] : 1;
     }
     
     
     //up
     for (int i=0; i<=h; i++)
     {  
        diff_up *= weight_data[channel + c];
        diff_up += top_up[((n*channel + c)*height + i)*width + w];
        diff_up *= (top_up_data[((n*channel + c)*height + i)*width + w]<=0)? slope_data[c] : 1;
     }

     bottom_diff[index] = diff_right + diff_left + diff_down + diff_up;

     //printf("ori_diff: %f, left: %f, up: %f, right: %f, down: %f, n: %d, c: %d, h:%d, w:%d\n",bottom_diff[index], diff_left, diff_up, diff_right, diff_down,n,c,h,w);
  }
}

template <typename Dtype>
__global__ void RATSlopeBackward(const int nthreads, const Dtype* top_left, const Dtype* top_up, const Dtype* top_right, const Dtype* top_down, const Dtype* top_left_data, const Dtype* top_up_data, const Dtype* top_right_data, const Dtype* top_down_data, const int channel, const int height, const int width, const Dtype* weight_data, const Dtype* bottom_data, Dtype* slope_map, const Dtype* slope_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  
  int w = index % width;
  int h = (index / width) % height;
  int c = (index / width / height) % channel;
  int n = index / width / height / channel;

  //left
  Dtype diff_temp = 0;
  float acc_weight = 1;
  float acc_slope = 1;
  int count_less_than_zero = 0;
  for (int m=w+1; m<width; m++)
  {  
      if (top_left_data[((n*channel + c)*height + h)*width + m]<0)
      { 
         count_less_than_zero++;
         diff_temp += count_less_than_zero*acc_slope*acc_weight*top_left[((n*channel + c)*height + h)*width + m];
  
         acc_slope *= slope_data[c];
      }

      acc_weight *= weight_data[c];
  }


  
  // up
  // diff_temp accumulate
  acc_weight = 1;
  acc_slope = 1;
  count_less_than_zero = 0;
  for (int m=h+1; m<height; m++)
  {  
      if (top_up_data[((n*channel + c)*height + m)*width + w]<0)
      { 
         count_less_than_zero++;
         diff_temp += count_less_than_zero*acc_slope*acc_weight*top_up[((n*channel + c)*height + m)*width + w];
  
         acc_slope *= slope_data[c];
      }

      acc_weight *= weight_data[channel + c];               
  }

  
  //right
  // diff_temp accumulate
  acc_weight = 1;
  acc_slope = 1;
  count_less_than_zero = 0;
  for (int m=w-1; m>=0; m--)
  {  
       if (top_right_data[((n*channel + c)*height + h)*width + m]<0)
       { 
          count_less_than_zero++;
          diff_temp += count_less_than_zero*acc_slope*acc_weight*top_right[((n*channel + c)*height + h)*width + m];
  
          acc_slope *= slope_data[c];
       }

       acc_weight *= weight_data[2*channel + c];
                     
  }
  
      

  //down
  // diff_temp accumulate
  acc_weight = 1;
  acc_slope = 1;
  count_less_than_zero = 0;
  for (int m=h-1; m>=0; m--)
  {  
      if (top_down_data[((n*channel + c)*height + m)*width + w]<0)
      { 
          count_less_than_zero++;
          diff_temp += count_less_than_zero*acc_slope*acc_weight*top_down[((n*channel + c)*height + m)*width + w];
  
          acc_slope *= slope_data[c];
      }

      acc_weight *= weight_data[3*channel + c];
                    
   }

   slope_map[((n*channel + c)*height + h)*width + w] = diff_temp*bottom_data[((n*channel + c)*height + h)*width + w];
 }  
}


template <typename Dtype>
__global__ void RATSlopeSetValueBackward(const int channel, int batch_num, const int height, const int width,  Dtype* slope_diff, Dtype* slope_map, const float lr_multi, const float max_gradient) {
  CUDA_KERNEL_LOOP(index, channel) {
  

  int c = index;

    float diff;
    
    diff = 0;
    for (int i=0; i<batch_num; i++)
    {
        for (int j=0; j<height; j++)
        {
            for (int k=0; k<width; k++)
            {
               
              diff += slope_map[((i*channel + c)*height + j)*width + k];
            }
        }
    }
  
    if (diff > max_gradient)
    { diff = max_gradient;}
    else if (diff < -max_gradient)
    { diff = -max_gradient;}

    slope_diff[c] = diff*lr_multi;

    //printf("The gradient is: %f\n", diff);
 }
}


template <typename Dtype>
void RATLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  int count = bottom[0]->count();
 
  Dtype* top_data[4];

  for (int i=0; i<4; i++)
  {
      top_data[i] = top[i]->mutable_gpu_data();
  }

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* weight_data = this->blobs_[0]->gpu_data();
  const Dtype* slope_data = this->blobs_[1]->gpu_data();

  RATForward<Dtype>  
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data[0], top_data[1], top_data[2], top_data[3], channels_, height_, width_, weight_data, slope_data);
  CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void RATLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = bottom[0]->count();
  const int dim = bottom[0]->count(2);
  const int channels = bottom[0]->channels();

  const Dtype* top_left = top[0]->gpu_diff();
  const Dtype* top_up = top[1]->gpu_diff();
  const Dtype* top_right = top[2]->gpu_diff();
  const Dtype* top_down = top[3]->gpu_diff();

  const Dtype* top_left_data = top[0]->gpu_data();
  const Dtype* top_up_data = top[1]->gpu_data();
  const Dtype* top_right_data = top[2]->gpu_data();
  const Dtype* top_down_data = top[3]->gpu_data();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* weight_data = this->blobs_[0]->gpu_data();
  const Dtype* slope_data = this->blobs_[1]->gpu_data();

  RATBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, top_left, top_up, top_right, top_down, top_left_data, top_up_data, top_right_data, top_down_data, channels_, height_, width_, weight_data, slope_data);
  CUDA_POST_KERNEL_CHECK;

  
  if (learnable_slope_)
  {
     Dtype* slope_diff = this->blobs_[1]->mutable_gpu_diff();
     
     Dtype* slope_map = slope_diff_map.mutable_gpu_data();
     const Dtype* bottom_data = bottom[0]->gpu_data();
     int batch_num = bottom[0]->shape(0);
     
    RATSlopeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_left, top_up, top_right, top_down, top_left_data, top_up_data, top_right_data, top_down_data, channels_, height_, width_, weight_data, bottom_data, slope_map, slope_data);
    CUDA_POST_KERNEL_CHECK;
   

    RATSlopeSetValueBackward<Dtype><<<CAFFE_GET_BLOCKS(channels_), CAFFE_CUDA_NUM_THREADS>>>(channels_, batch_num, height_, width_, slope_diff, slope_map, lr_multi_, max_gradient_);
    CUDA_POST_KERNEL_CHECK;
  }

  
  if (!weight_fixed) //TO DO
  {
    NOT_IMPLEMENTED;  
    /*
    Dtype* weight_map = weight_diff_map.mutable_gpu_data();

    int cdim = 4*channels_;
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    int batch_num = bottom[0]->shape(0);


    RATParamBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_left, top_up, top_right, top_down, top_left_data, top_up_data, top_right_data, top_down_data, channels_, height_, width_, weight_data, bottom_data, weight_map);
    CUDA_POST_KERNEL_CHECK;

    RATParamSetValueBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim), CAFFE_CUDA_NUM_THREADS>>>(cdim, batch_num, channels_, height_, width_, weight_diff, weight_map);
    CUDA_POST_KERNEL_CHECK;
    */
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RATLayer);

}  // namespace caffe
