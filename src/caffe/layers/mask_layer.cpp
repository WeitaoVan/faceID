#include <cfloat>
#include <vector>

#include "caffe/layers/mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
/*
*		Created by Weitao Wan
*		Perform spatial mask weighting on bottom[0]
*		(TODO):This cpp version is NOT implemented.
*/
template <typename Dtype>
void MaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[1]->channels()== 1) << 
	"bottom[1] is supposed to be the mask which has 1 channel.";
  CHECK(bottom[0]->width() == bottom[1]->width()
	&& bottom[0]->height() == bottom[1]->height()) 
	<<"spatial dimensions of 2 bottoms mismatch ";
  scale = this->layer_param_.mask_param().scale();
 }

template <typename Dtype>
void MaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  mask.ReshapeLike(*bottom[0]);
  tmp.ReshapeLike(*bottom[0]);
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "cpp version not implemented";
}

template <typename Dtype>
void MaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	LOG(FATAL) << "cpp version is not implemented";
}

#ifdef CPU_ONLY
STUB_GPU(MaskLayer);
#endif

INSTANTIATE_CLASS(MaskLayer);
REGISTER_LAYER_CLASS(Mask);

}  // namespace caffe
