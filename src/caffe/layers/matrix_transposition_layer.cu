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


template <typename Dtype>  //left up right down
__global__ void MatrixTForward(const int nthreads, const Dtype* bottom_data, Dtype* top_data, int M, int N) {
  CUDA_KERNEL_LOOP(index, nthreads) {

  int x_hw = index % N;      // x_hw = z_c
  int x_c = (index / N) % M; // x_c = z_hw
  int n = index / N / M;
 
  // bottom_data: n*M*N; top_data: x*N*M
  top_data[(n*N+x_hw)*M+x_c] = bottom_data[index]; //top_data[(n*N+z_c)*M+z_hw]
  }
}

template <typename Dtype>
void MatrixTranspositionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* X_data = bottom[0]->gpu_data();
  Dtype* Z_data = top[0]->mutable_gpu_data();
  
  int count = bottom[0]->count();

  MatrixTForward<Dtype>  
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, X_data, Z_data, M_, N_);
  CUDA_POST_KERNEL_CHECK;
  
}


template <typename Dtype>  //left up right down
__global__ void MatrixTBackward(const int nthreads, Dtype* bottom_diff, const Dtype* top_diff, int M, int N) {
  CUDA_KERNEL_LOOP(index, nthreads) {

  int x_hw = index % N;      // x_hw = z_c
  int x_c = (index / N) % M; // x_c = z_hw
  int n = index / N / M;
 
  // bottom_diff: n*M*N; top_diff: x*N*M
  bottom_diff[index] = top_diff[(n*N+x_hw)*M+x_c];  //top_data[(n*N+z_c)*M+z_hw]
  }
}

template <typename Dtype>
void MatrixTranspositionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* Z_diff = top[0]->gpu_diff();
  Dtype* X_diff = bottom[0]->mutable_gpu_diff();

  int count = bottom[0]->count();
 
    MatrixTBackward<Dtype>  
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, X_diff, Z_diff, M_, N_);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(MatrixTranspositionLayer);

}  // namespace caffe
