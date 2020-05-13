// ------------------------------------------------------------------
// Copyright (c) 2018
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
// ------------------------------------------------------------------

#include <vector>
#include <iostream>

#include "caffe/filler.hpp"

#include "caffe/layers/repeat_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void RepeatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const int num_axes = bottom[0]->num_axes();
  CHECK(num_axes == 3)
       << "The input shape should be [n,c,m], m=h*w.";

  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();

  RepeatParameter repeat_parameter = this->layer_param_.repeat_param();
  width_ = (repeat_parameter.repeat_num()==-1) ? height_ : repeat_parameter.repeat_num();

}

template <typename Dtype>
void RepeatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  top[0]->Reshape(bottom[0]->shape(0), channels_, height_, width_);

}

template <typename Dtype>
void RepeatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  NOT_IMPLEMENTED;
  
}

template <typename Dtype>
void RepeatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RepeatLayer);
#endif

INSTANTIATE_CLASS(RepeatLayer);
REGISTER_LAYER_CLASS(Repeat);

}  // namespace caffe
