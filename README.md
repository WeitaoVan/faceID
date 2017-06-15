# Deep Face Recognition with Caffe Implementation

This branch is developed for face recognition with occlusion, the related paper is as follows.
    
    OCCLUSION ROBUST FACE RECOGNITION BASED ON MASK LEARNING[C]
    Weitao, Wan and Jiansheng, Chen
    2017 IEEE International Conference on Image Processing (ICIP)

* [Architecture](#architecture)
* [Results](#results)
* [Files](#files)
* [Train_Model](#train_model)
* [Contact](#contact)
* [Citation](#citation)
* [LICENSE](#license)
* [README_Caffe](#readme_caffe)

### Architecture
Our network architecture 

![Picture](https://github.com/WeitaoVan/faceID/blob/MaskNet/image/structure.jpg)

### Results
The generated masks on faces with occlusion

![Picture](https://github.com/WeitaoVan/faceID/blob/MaskNet/image/mask.jpg)

Face verification on lfw validation set with synthesized square blocks with varying sizes.

![Picture](https://github.com/WeitaoVan/faceID/blob/MaskNet/image/Verification_Acc.jpg)

### Files
- Caffe with center loss imported from https://github.com/ydwen/caffe-face
- Mask Layer
  * src/caffe/layers/mask_layer.cu
  * src/caffe/layers/mask_layer.cpp (CPU mode not supported for now)
  * include/caffe/layers/mask_layer.hpp
- training network
  * face_example/train_centerMask2Pool2_ori.prototxt
  * face_example/face_solver.prototxt

### Train_Model
1. Specify your mask size in 'num_output' (the value should equal to height x width of the mask)

		
		layer {
		  name: "mask_ip"
		  type: "InnerProduct"
		  bottom: "mask_conv3"
		  top: "mask_ip"
		  param {
		    name: "mask_ip_w"
		    lr_mult: 1
		  }
		  param {
		    name: "mask_ip_b"
		    lr_mult: 2
		  }
		  inner_product_param {
		    num_output: 572     # change it based on your network
		    weight_filler {
		      type: "constant"
		    }
		    bias_filler {
		      type: "constant"
		    }
		  }
		}
		
2. Choose the location to insert the mask layer. I placed it after 'pool2'.

		
		layer {
		  name: "mask"
		  type: "Mask"
		  bottom: "pool2"
		  bottom: "mask_2d"
		  top: "masked_pool2"
		  mask_param {
		    scale: 1
		  }
		}
		

### Contact 
Weitao Wan(wwt16@mails.tsinghua.edu.cn)

### Citation
Please consider citing the following paper if it helps your research. 

    @inproceedings{wan2017mask,
      title={OCCLUSION ROBUST FACE RECOGNITION BASED ON MASK LEARNING},
      author={Weitao, Wan and Jiansheng, Chen},
      booktitle={IEEE International Conference on Image Processing (ICIP)},
      pages={},
      year={2017},
      organization={IEEE}
    }

### License
Copyright (c) Weitao Wan

All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

***

### README_Caffe
# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
