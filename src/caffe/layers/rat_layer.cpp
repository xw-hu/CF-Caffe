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
#include <iostream>

#include "caffe/filler.hpp"

#include "caffe/layers/rat_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void RATLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  RATParameter rat_parameter = this->layer_param_.rat_param();
  weight_fixed = rat_parameter.weight_fixed();
  initial_value_ = rat_parameter.initial_value();
  slope_ = rat_parameter.slope();
  learnable_slope_ = rat_parameter.learnable_slope();
  lr_multi_ = rat_parameter.lr_multi();
  max_gradient_ = rat_parameter.max_gradient();
   
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else { 
    this->blobs_.resize(2);
    this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels_*4)));

    this->blobs_[1].reset(new Blob<Dtype>(vector<int>(1, channels_)));
  }
  
  shared_ptr<Filler<Dtype> > filler;
 
  FillerParameter filler_param;
  filler_param.set_type("constant");
  filler_param.set_value(initial_value_);
  filler.reset(GetFiller<Dtype>(filler_param));
  filler->Fill(this->blobs_[0].get());  

  FillerParameter filler_slope;
  filler_slope.set_type("constant");
  filler_slope.set_value(slope_);
  filler.reset(GetFiller<Dtype>(filler_slope));
  filler->Fill(this->blobs_[1].get());  


  // Propagate gradients to the parameters (as directed by backward pass).
  if (!weight_fixed && learnable_slope_)
     this->param_propagate_down_.resize(this->blobs_.size(), true);
  else if (!weight_fixed || learnable_slope_)
     this->param_propagate_down_.resize(1, true);
 
}

template <typename Dtype>
void RATLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  for (int i=0;i<4;i++) //left up right down
  {
      top[i]->Reshape(bottom[0]->shape(0), channels_, height_, width_);
  }

  if (learnable_slope_)
     slope_diff_map.Reshape(bottom[0]->shape(0), channels_, height_, width_);
}

template <typename Dtype>
void RATLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  NOT_IMPLEMENTED;
  
  //Dtype* top_data[4];
  //const Dtype* bottom_data[1];

  //for (int i=0; i<4; i++)
  //{
  //    top_data[i] = top[i]->mutable_cpu_data();
  //}
  //bottom_data[0] = bottom[0]->cpu_data();
 
}

template <typename Dtype>
void RATLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RATLayer);
#endif

INSTANTIATE_CLASS(RATLayer);
REGISTER_LAYER_CLASS(RAT);

}  // namespace caffe
