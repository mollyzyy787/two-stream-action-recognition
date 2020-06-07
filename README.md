# two-stream-action-recognition-using-pose-flow
I used a spatial and motion stream cnn with ResNet18 for modeling video information in UCF101 dataset.I have chosen 15 action classes (a subset of UCF101) to train the two-stream network. The 15 selected classes are specified in UCF_list/classInd15.txt
## Reference Paper
*  [[1] Two-stream convolutional networks for action recognition in videos](http://papers.nips.cc/paper/5353-two-stream-convolutional)
*  [[2] Temporal Segment Networks: Towards Good Practices for Deep Action Recognition](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_2)
* [[3] TS-LSTM and Temporal-Inception: Exploiting Spatiotemporal Dynamics for Activity Recognition](https://arxiv.org/abs/1703.10667)
* [[4] Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/abs/1611.08050)

## 1. Data
  ### 1.1 Spatial input data -> rgb frames
  * We extract RGB frames from each video in UCF101 dataset with sampling rate: 10 and save as .jpg image in disk which cost about 5.9G.
  ### 1.2 Motion input data -> stacked pose flow images
  Process the image sequences use demo/process_pose.py in this git [repo](https://github.com/mollyzyy787/pytorch_Multi-Person_Pose_Flow_Estimation.git). Please modify relevant paths.
  ### 1.3 (Alternative)Download the preprocessed data directly from [feichtenhofer/twostreamfusion](https://github.com/feichtenhofer/twostreamfusion))
  * RGB images
  ```
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002
  wget http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003
  
  cat ucf101_jpegs_256.zip* > ucf101_jpegs_256.zip
  unzip ucf101_jpegs_256.zip
  ```
  

## 2. Model
  ### 2.1 Spatial cnn
  * As mention before, we use ResNet18 first pre-trained with ImageNet then fine-tuning on our UCF101 spatial rgb image dataset. 
  ### 2.2 Motion cnn
  * Input data of motion cnn is a stack of rgb pose flow images. So it's input shape is (30, 224, 224) which can be considered as a 30-channel image. 
  * In order to utilize ImageNet pre-trained weight on our model, we have to modify the weights of the first convolution layer pre-trained  with ImageNet from (64, 3, 7, 7) to (64, 30, 7, 7). 
  * In [2] Wang provide a method called **Cross modality pre-
  
  ** to do such weights shape transform. He first average the weight value across the RGB channels and replicate this average by the channel number of motion stream input( which is 30 is this case)
  
## 3. Training strategies
  ###  3.1 Spatial cnn
  * Here we utilize the techniques in Temporal Segment Network. For every videos in a mini-batch, we randomly select 3 frames from each video. Then a consensus among the frames will be derived as the video-level prediction for calculating loss.
  ### 3.2 Motion cnn
  * In every mini-batch, we randomly select 64 (batch size) videos from 9537 training videos and futher randomly select 1 stacked optical flow in each video. 
  ### 3.3 Data augmentation
  * Both stream apply the same data augmentation technique such as random cropping.
## 4. Testing method
  * For every 3783 testing videos, we uniformly sample 19 frames in each video and the video level prediction is the voting result of all 19 frame level predictions.
  * The reason we choose the number 19 is that the minimun number of video frames in UCF101 is 28 and we have to make sure there are sufficient frames for testing in 10 stack motion stream.
## 5. Performance
<p align="left">
<img src="https://github.com/mollyzyy787/two-stream-action-recognition-using-pose-flow/conf_matrix.png", width="720">
</p>
## 6. Testing on Your Device
  ### Spatial stream
 * Please modify path to the UCF101 dataset on your device.
 * Training and testing
 ```
 python spatial_cnn_15.py --resume PATH_TO_PRETRAINED_MODEL
 ```
 * Only testing
 ```
 python spatial_cnn_15.py --resume PATH_TO_PRETRAINED_MODEL --evaluate
 ```
 
 ### Motion stream
 *  Please modify path to the UCF101 dataset on your device.
  * Training and testing
 ```
 python motion_pose_cnn_15.py --resume PATH_TO_PRETRAINED_MODEL
 ```
 * Only testing
 ```
 python motion_pose_cnn_15.py --resume PATH_TO_PRETRAINED_MODEL --evaluate
 ```
 

