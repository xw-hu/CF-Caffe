// ------------------------------------------------------------------
// Copyright (c) 2018
// The Chinese University of Hong Kong
// Written by Hu Xiaowei
//
// Matrix Transposition (n,c,mn) -> (n,mn,c) 
// ------------------------------------------------------------------

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/matrix_transposition_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MatrixTranspositionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Reshape(bottom, top);
}

template <typename Dtype>
void MatrixTranspositionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->num_axes()==3)
    << "X blob must be of shape (B,M,N)!";
  const bool X_nobatch = false;
  const int Bx = X_nobatch ? 1 : bottom[0]->shape(0); //get the batch num of X

  const int Rx = bottom[0]->shape(1-int(X_nobatch));
  const int Cx = bottom[0]->shape(2-int(X_nobatch));
  M_ = Rx;
  N_ = Cx;
  
  const bool Z_nobatch = X_nobatch;
  vector<int> top_shape(Z_nobatch ? 2 : 3);
  if (Z_nobatch) {
    top_shape[0]=N_;
    top_shape[1]=M_;
  } else {
    top_shape[0]=Bx;
    top_shape[1]=N_;
    top_shape[2]=M_;
  }
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MatrixTranspositionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MatrixTranspositionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

 NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MatrixTranspositionLayer);
#endif

INSTANTIATE_CLASS(MatrixTranspositionLayer);
REGISTER_LAYER_CLASS(MatrixTransposition);

}  // namespace caffe
