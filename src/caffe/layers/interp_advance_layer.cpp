#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/layers/interp_advance_layer.hpp"

namespace caffe {

template <typename Dtype>
void InterpAdvanceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  pad_beg_ = 0;
  pad_end_ = 0;
}

template <typename Dtype>
void InterpAdvanceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  height_in_eff_ = height_in_;
  width_in_eff_ = width_in_;

  height_out_ = bottom[1]->height();
  width_out_ = bottom[1]->width();

  CHECK_GT(height_in_eff_, 0) << "height should be positive";
  CHECK_GT(width_in_eff_, 0) << "width should be positive";
  CHECK_GT(height_out_, 0) << "height should be positive";
  CHECK_GT(width_out_, 0) << "width should be positive";
  top[0]->Reshape(num_, channels_, height_out_, width_out_);
}

template <typename Dtype>
void InterpAdvanceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_cpu_interp2<Dtype,false>(num_ * channels_,
    bottom[0]->cpu_data(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->mutable_cpu_data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

template <typename Dtype>
void InterpAdvanceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  caffe_cpu_interp2_backward<Dtype,false>(num_ * channels_,
    bottom[0]->mutable_cpu_diff(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->cpu_diff(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

#ifndef CPU_ONLY
template <typename Dtype>
void InterpAdvanceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_gpu_interp2<Dtype,false>(num_ * channels_,
    bottom[0]->gpu_data(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->mutable_gpu_data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

template <typename Dtype>
void InterpAdvanceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  caffe_gpu_interp2_backward<Dtype,false>(num_ * channels_,
    bottom[0]->mutable_gpu_diff(), - pad_beg_, - pad_beg_, height_in_eff_, width_in_eff_, height_in_, width_in_,
    top[0]->gpu_diff(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}
#endif

#ifdef CPU_ONLY
STUB_GPU(InterpAdvanceLayer);
#endif

INSTANTIATE_CLASS(InterpAdvanceLayer);
REGISTER_LAYER_CLASS(InterpAdvance);

}  // namespace caffe
