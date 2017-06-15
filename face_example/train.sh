/home/wwt/caffe_centerFace/build/tools/caffe train -solver /home/wwt/caffe_centerFace/face_example/face_solver.prototxt \
2>&1 | tee log/train_angle_share.log
#						   -weights /media/wwt/860G/model/face/centerMask2Pool3_iter_2000.caffemodel \
#-weights /media/wwt/860G/model/face/center_simOcc_iter_33000.caffemodel \
#						   -weights /media/wwt/860G/model/face/fetch_fc6_centerloss_iter_1.caffemodel \
