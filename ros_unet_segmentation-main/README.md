# ros_unet_segmentation
this is a ROS implementation for semantic segmentation with the model Unet based on the vigne dataset or outdoor dataset.
## Environnement
    Ubuntu 18.04.6 LTS
    ROS Melodic
    Pytorch
## requirement
    pip3 install torch torchvision torchaudio
    pip3 install opencv-python
    pip3 install pyaml
    pip3 install rospkg
    pip3 install -q segmentation-models-pytorch 
    pip3 install pyrealsense2
    
## workspace:
  ### create a catkin workspace 
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/
    #Add workspace to bashrc.
    echo 'source ~/catkin_ws/devel/setup.bash' >> ~/.bashrc
    #Clone repo
    cd ~/catkin_ws/src
    git clone https://github.com/robotgogo-seg/ros_unet_segmentation.git
    sudo chmod +x ros_unet_segmentation/scripts/unet_ros_inference
    #build
    cd ..
    catkin_make
    #refresh workspace
    source ~/catkin_ws/devel/setup.bash
## Test
### change launch file 
  topic_subscriber: topic of the rosbag 
  device: "cuda:0" or "cpu"
  cfg_file_path: path to the configuration file
  model_ckpt_path: path to the pretrained model
put the pretrained models on the folder 'src/ros_unet_segmentation/ckpt'
### test the model
  roslaunch ros_unet_segmentation unet_segmentation.launch
### play rosbag
    #launch realsense package for ROS
    roslaunch realsense2_camera rs_camera.launch
    #open RVIZ and add topic for image 
    rviz
    #to play the rosbag
    rosbag  play UPJVOutdoor_person_3.bag
