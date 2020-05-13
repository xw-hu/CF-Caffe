// ------------------------------------------------------------------
// Copyright (c) 2018
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
//
// recurrently-attenuating translation
// y(i,j) = alpha*f(i-1,j)+f(i,j)
// if y(i,j)>=0, y(i,j)=y(i,j) else y(i,j)=beta*y(i,j)
// ------------------------------------------------------------------

#ifndef CAFFE_RAT_LAYER_HPP_
#define CAFFE_RAT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/bias_layer.hpp"

namespace caffe {

template <typename Dtype>
class RATLayer : public Layer<Dtype> {
 public:
  explicit RATLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RAT"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }  
  virtual inline int ExactNumTopBlobs() const { return 4; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool weight_fixed;
  float initial_value_;
  float slope_;
  int channels_;
  int height_;
  int width_;
  Blob<Dtype> slope_diff_map;
  bool learnable_slope_;
  float lr_multi_;
  float max_gradient_;
};


}  // namespace caffe

#endif  // CAFFE_RAT_LAYER_HPP_
