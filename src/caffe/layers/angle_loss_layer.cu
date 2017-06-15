#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/angle_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Compute_loss_gpu(const int M, const Dtype* dotxb, const Dtype* dotx, const Dtype* dotb, 
				 Dtype* loss) {
  CUDA_KERNEL_LOOP(index, M) {
    loss[index] = Dtype(1.0) - dotxb[index] * dotxb[index] / (dotx[index] * dotb[index] + Dtype(1e-8)); 
  }
}

template <typename Dtype>
__global__ void Compute_center_diff_gpu(int nthreads, const int M, const int K, const Dtype* x,
        const Dtype* b, const Dtype* label, const Dtype* dotxb, const Dtype* dotx, const Dtype* dotb,
        Dtype* center_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int count = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (label_value == index) {
        count++;
	Dtype D = dotx[m];
        for (int k = 0; k < K; k++) {
          int idx_x = m * K + k;
	  int idx_b = index * K + k;
    	  Dtype x_val = x[idx_x];
          Dtype b_val = b[idx_b];
    	  Dtype cross_term = dotxb[m] - x_val * b_val;
    	  Dtype A = x_val * x_val;
    	  Dtype B = 2 * x_val * cross_term;
    	  Dtype C = cross_term * cross_term;
    	  Dtype E = dotx[m] * (dotb[m] - b_val * b_val);
          center_diff[idx_b] += Dtype(-1.0) * ((Dtype(-1.0)*D*B*b_val*b_val + 2*(A*E - C*D)*b_val + B*E)
				                    / (dotx[m] * dotx[m] * dotb[m] * pow(dotb[m], Dtype(0.25))  + 1e-8));
        }
      }
    }
    if(count > 1){
    	for (int k = 0; k < K; k++) {
     	  center_diff[index * K + k] = center_diff[index * K + k] / Dtype(count);
    	}
    }
  }
}

template <typename Dtype>
__global__ void Compute_bottom_diff_gpu(int nthreads, int K, const Dtype* x,
        const Dtype* b, const Dtype* label, const Dtype* dotxb, const Dtype* dotx, const Dtype* dotb,
        Dtype* bottom_diff){
  CUDA_KERNEL_LOOP(index, nthreads){
    int m = index / K;
    int k = index % K;
    int label_value = static_cast<int>(label[m]);
    Dtype x_val = x[index];
    Dtype b_val = b[label_value * K + k];
    Dtype cross_term = dotxb[m] - x_val * b_val;
    Dtype A = b_val * b_val;
    Dtype B = 2 * b_val * cross_term;
    Dtype C = cross_term * cross_term;
    Dtype D = dotb[m];
    Dtype E = dotb[m] * (dotx[m] - x_val * x_val);
    bottom_diff[index] = Dtype(-1.0) * ((Dtype(-1.0)*D*B*x_val*x_val + 2*(A*E - C*D)*x_val + B*E)
                            / (dotx[m] * pow(dotx[m], Dtype(0.25)) * dotb[m] * dotb[m] + 1e-8));
  }
}


template <typename Dtype>
void AngleLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* x = bottom[0]->gpu_data();
  const Dtype* b = this->blobs_[0]->gpu_data();
  Dtype mean_x_sq = 0;
  for(int i = 0; i < M_; i++){
	const int label_value = static_cast<int>(bottom[1]->cpu_data()[i]);
	const Dtype* px = &x[i * K_];
	const Dtype* pb = &b[label_value * K_];
	Dtype tmp;
	caffe_gpu_dot(K_, px, pb, &tmp);
	if(0){
		LOG(INFO) << "dotxb[" << i << "]=" << tmp;
	}
	dotxb.mutable_cpu_data()[i] = tmp;
	caffe_gpu_dot(K_, px, px, &tmp);
	if(0){
		//LOG(INFO) << "dotx = " << tmp;
		mean_x_sq += tmp;
	}
	dotx.mutable_cpu_data()[i] = tmp;
	caffe_gpu_dot(K_, pb, pb, &tmp);
	dotb.mutable_cpu_data()[i] = tmp;
	if(0){
		LOG(INFO) << "dotb=" << tmp;
		LOG(INFO) << " ";
	}
  }
  if(0){
	LOG(INFO) << "mean of dotx=" << mean_x_sq / Dtype(M_);
  }
  Compute_loss_gpu<Dtype><<<CAFFE_GET_BLOCKS(M_),
      CAFFE_CUDA_NUM_THREADS>>>(M_, dotxb.gpu_data(), dotx.gpu_data(), 
				dotb.gpu_data(), loss_data.mutable_gpu_data());
  Dtype loss = 0.0;
  caffe_gpu_asum(M_, loss_data.gpu_data(), &loss);
  loss = loss / M_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void AngleLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  int nthreads = N_;
  const Dtype* x = bottom[0]->gpu_data();
  const Dtype* b = this->blobs_[0]->gpu_data();
  Dtype* blob_diff = this->blobs_[0]->mutable_gpu_diff();
  caffe_gpu_set(N_ * K_, (Dtype)0., blob_diff);
  // diff for center parameters (i.e. b)
  bool diff_to_center = false;
  if(diff_to_center){
  Compute_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_, x, b, bottom[1]->gpu_data(), dotxb.gpu_data(), dotx.gpu_data(),
                                dotb.gpu_data(), blob_diff);
  caffe_gpu_scal(N_ * K_, Dtype(0.1), blob_diff);
  }
  if (propagate_down[0]) {
  // diff for bottom[0]
    Compute_bottom_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(M_ * K_),
        CAFFE_CUDA_NUM_THREADS>>>(M_ * K_, K_, x, b, bottom[1]->gpu_data(), dotxb.gpu_data(), dotx.gpu_data(),
                                  dotb.gpu_data(), bottom[0]->mutable_gpu_diff());
    caffe_gpu_scal(M_ * K_, top[0]->cpu_diff()[0] / M_, bottom[0]->mutable_gpu_diff());
    if(0){ //DEBUG
	for(int i = 0; i < 4; i++){
		LOG(INFO) << "bottom_diff[" << i << "]=" << bottom[0]->cpu_diff()[i];
	}
	int label_value = bottom[1]->cpu_data()[0];
	for(int i = 0; i < 0; i++){
		LOG(INFO) << "center_diff[" << i << "]=" << this->blobs_[0]->cpu_diff()[label_value * K_ + i];
	}
    }
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AngleLossLayer);

}  // namespace caffe
