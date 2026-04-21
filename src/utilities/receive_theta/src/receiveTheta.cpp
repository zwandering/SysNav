#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/time.hpp"
#include "rclcpp/clock.hpp"
#include "builtin_interfaces/msg/time.hpp"

#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <cv_bridge/cv_bridge.hpp>

using namespace std;
using namespace cv;

Mat image, imageSmall;

string imuTopicName = "/imu/data";
int topBottonMargin = 160;
double imageLatency = 0;
int compQuality = 50;
bool alwaysPubCompImage = false;
bool showImage = false;

double systemToImuTime = 0;

cv_bridge::CvImage bridge;

rclcpp::Node::SharedPtr nh;

void imuHandler(const sensor_msgs::msg::Imu::ConstSharedPtr imuIn)
{
  systemToImuTime = rclcpp::Time(imuIn->header.stamp).seconds() - nh->now().seconds();
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  nh = rclcpp::Node::make_shared("receiveTheta");

  nh->declare_parameter<string>("imuTopicName", imuTopicName);
  nh->declare_parameter<int>("topBottonMargin", topBottonMargin);
  nh->declare_parameter<double>("imageLatency", imageLatency);
  nh->declare_parameter<int>("compQuality", compQuality);
  nh->declare_parameter<bool>("alwaysPubCompImage", alwaysPubCompImage);
  nh->declare_parameter<bool>("showImage", showImage);

  nh->get_parameter("imuTopicName", imuTopicName);
  nh->get_parameter("topBottonMargin", topBottonMargin);
  nh->get_parameter("imageLatency", imageLatency);
  nh->get_parameter("compQuality", compQuality);
  nh->get_parameter("alwaysPubCompImage", alwaysPubCompImage);
  nh->get_parameter("showImage", showImage);

  auto imuSub = nh->create_subscription<sensor_msgs::msg::Imu>(imuTopicName, 50, imuHandler);

  auto imagePub = nh->create_publisher<sensor_msgs::msg::Image>("/camera/image", 2);

  auto compressedPub = nh->create_publisher<sensor_msgs::msg::CompressedImage>("/camera/image/compressed", 2);

  VideoCapture video("thetauvcsrc ! decodebin ! videoconvert ! appsink");
  if(!video.isOpened()) {
    printf ("\nCannot open device, exit.\n\n");
    return 0;
  }

  std::vector<int> compression_params;
  compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
  compression_params.push_back(compQuality);

  while (rclcpp::ok()) {
    video >> image;

    double imageTime = nh->now().seconds() + systemToImuTime - imageLatency;

    if (topBottonMargin < 0) topBottonMargin = 0;
    else if (topBottonMargin > image.rows / 2 - 1) topBottonMargin = image.rows / 2 - 1;

    if (topBottonMargin > 0) {
      Rect roi = Rect(0, topBottonMargin, image.cols, image.rows - 2 * topBottonMargin);
      image = image(roi);
    }

    std_msgs::msg::Header header;
    header.frame_id = "camera";
    header.stamp = rclcpp::Time(static_cast<uint64_t>(imageTime * 1e9));
    sensor_msgs::msg::Image::SharedPtr imagePtr = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();
    imagePub->publish(*imagePtr);

    if (compressedPub->get_subscription_count() > 0 || alwaysPubCompImage) {
      std::vector<uint8_t> jpeg_image;
      cv::imencode(".jpg", image, jpeg_image, compression_params);
      sensor_msgs::msg::CompressedImage compressed_msg;
      compressed_msg.header = header;
      compressed_msg.format = "jpeg";
      compressed_msg.data = jpeg_image;
      compressedPub->publish(compressed_msg);
    }

    if (showImage) {
      resize(image, imageSmall, Size(image.cols / 2, image.rows / 2));
      imshow("Image (50% res)", imageSmall);
    }

    rclcpp::spin_some(nh);
    waitKey(10);
  }

  return 0;
}
