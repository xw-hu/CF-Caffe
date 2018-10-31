GLOG_logtostderr=1 ../../../build/tools/caffe train \
  --solver=solver.prototxt \
  --weights=../../../models/ResNet-101-model.caffemodel \
  --gpu=0  2>&1 | tee log.log

