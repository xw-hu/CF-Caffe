// ------------------------------------------------------------------
// Copyright (c) 2018
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
// ------------------------------------------------------------------

#include <vector>

#include "caffe/layers/repeat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>  //left up right down
__global__ void RepeatForward(const int nthreads, const Dtype* bottom_data, Dtype* top_data, const int channel, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {

     //int w = index % width;
     int h = (index / width) % height;
     int c = (index / width / height) % channel;
     int n = index / width / height / channel;

     
     top_data[index] = bottom_data[(n*channel + c)*height + h]; 
     
  }
}

template <typename Dtype>
__global__ void RepeatBackward(const int nthreads, Dtype* bottom_diff, const Dtype* top_diff, const int channel, const int height, const int width) {
  CUDA_KERNEL_LOOP(index, nthreads) {
     
     // nthreads is the total number of bottom[0] (width_bottom=1)
     // global variable width is the width for top[0]

     int h = (index) % height;
     int c = (index / height) % channel;
     int n = index / height / channel;

     Dtype diff_acc = 0;

     for (int i=0; i<width; i++)
     {  
        diff_acc += top_diff[((n*channel + c)*height + h)*width + i];
        //printf("index = %d, i = %d, c= %d, channel= %d, top_diff: %f, diff_acc is: %f\n", index, i, c, channel, top_diff[((n*channel + c)*height + h)*width + i], diff_acc);
     }
     bottom_diff[index] = diff_acc;
     //printf("bottom_diff is: %f\n", bottom_diff[index]);
  }
}


template <typename Dtype>
void RepeatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  int count = top[0]->count();
 
  Dtype* top_data = top[0]->mutable_gpu_data();

  const Dtype* bottom_data = bottom[0]->gpu_data();

  RepeatForward<Dtype>  
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_data, channels_, height_, width_);
  CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
void RepeatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  int count = bottom[0]->count();

  const Dtype* top_diff= top[0]->gpu_diff();

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  RepeatBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_diff, top_diff, channels_, height_, width_);
  CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(RepeatLayer);

}  // namespace caffe
