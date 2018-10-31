#ifndef CAFFE_MATRIX_MULTIPLICATION_YT_LAYER_HPP_
#define CAFFE_MATRIX_MULTIPLICATION_YT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief compute matrix multiplication of two input X and Y and output $Z=XY^T$
 *
 * Input:
 * X: <BxMxK> or <MxK>
 * Y: <BxNxK> or <NxK>
 * Output:
 * Z: <BxMxN> or <MxN>
 *
 * If X shape is <BxMxK> while Y shape is <NxK>, then $Z=\{X_0*Y^T, X_1*Y^T, ..., X_{B-1}*Y^T\}$ by broadcasting Y.
 * And similar for the other case by broadcasting X.
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class MatrixMultiplicationYtLayer : public Layer<Dtype> {
 public:
  explicit MatrixMultiplicationYtLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MatrixMultiplicationYt"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_; //X <MxK>, Y<NxK>, Z<MxN>
  int K_;
  int N_;
};

}  // namespace caffe

#endif  // CAFFE_MATRIX_MULTIPLICATION_YT_LAYER_HPP_
