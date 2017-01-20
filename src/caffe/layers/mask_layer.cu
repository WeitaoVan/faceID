#include <cfloat>
#include <vector>

#include "caffe/layers/mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaskCopyGPU(const int count, const int num, const int channels,
			      const Dtype*  bottom_data, Dtype* mask){
	const int spatial_dim = count / num / channels;
	const int dim = spatial_dim * channels;
	CUDA_KERNEL_LOOP(index, count){
		const int bottom_idx = (index / dim) * spatial_dim
					+ (index % spatial_dim);
		mask[index] = bottom_data[bottom_idx];
	}
}

template <typename Dtype>
void MaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = top[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  
  if(0){ //DEBUG
    for(int i = 0; i < height * width; i++)
	LOG(INFO) << "mask2d[" << i << "] = " << bottom[1]->cpu_data()[i];
  }
  
  // use the bottom[1] Nx1xHxW mask to create the NxCxHxW mask, by duplicate each HxW map for C times.
  MaskCopyGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, 
			num, channels, bottom[1]->gpu_data(), mask.mutable_gpu_data());
  // scale [default = 1.0]
  caffe_gpu_scale(count, Dtype(scale), mask.gpu_data(), mask.mutable_gpu_data());

  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_mul(count, bottom[0]->gpu_data(), mask.gpu_data(), top_data);
   
  if(0){ //DEBUG
    for(int i = width * 6; i < width * 7; i++)
	if(mask.cpu_data()[i] < Dtype(0.18))
	LOG(INFO) << "mask[" << i << "] = " << mask.cpu_data()[i];
  }
}


template <typename Dtype>
__global__ void MaskBackwardGPU(const int count, const int num, const int channels,
	             const Dtype* tmp, Dtype* bottom_diff){
	const int spatial_dim = count / num / channels;
	const int dim = spatial_dim * channels;
	CUDA_KERNEL_LOOP(index, count){
		const int bottom_idx = (index / dim) * spatial_dim
					+ (index % spatial_dim);
		bottom_diff[bottom_idx] += tmp[index];
	}
}


template <typename Dtype>
void MaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = top[0]->count();
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const Dtype* top_diff = top[0]->gpu_diff();
  for (int i = 0; i < bottom.size(); ++i) {
    if (propagate_down[i]) {
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
	if(i == 0){ // diff for bottom feature map
		caffe_gpu_mul(count, mask.gpu_data(), top_diff, bottom_diff);
	} else{ // diff for bottom mask
		caffe_gpu_set(bottom[i]->count(), Dtype(0), bottom_diff);
		// multiply top_diff and feature maps element-wisely. (1)
		caffe_gpu_mul(count, bottom[0]->gpu_data(), top_diff, tmp.mutable_gpu_data());
		// scale [defaut = 1.0]
		caffe_gpu_scale(count, Dtype(scale), tmp.gpu_data(), tmp.mutable_gpu_data());
		// accumulate the multiplication values in (1) along the channel dimension
		MaskBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, 
				num, channels, tmp.gpu_data(), bottom_diff);
		if(0){  //DEBUG
			for(int k = 0; k < height * width / 13; k++){
				LOG(INFO) << "mask_diff[" << k << "]=" << bottom[1]->cpu_diff()[k];	
				//LOG(INFO) << "tmp[" << k << "]=" << tmp.cpu_data()[k];
				//LOG(INFO) << "tmp[" << k + height * width << "]=" << tmp.cpu_data()[k+height*width];
			}
		}
		
	}//if(i == 0)
    }//if(propagate_down[0])
  }//for
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskLayer);

}  // namespace caffe
