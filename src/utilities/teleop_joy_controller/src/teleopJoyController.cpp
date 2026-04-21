#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "rclcpp/rclcpp.hpp"

#include <sensor_msgs/msg/joy.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include "serial/serial.h"

using namespace std;

const double PI = 3.1415926;

string serialPort = "/dev/ttyACM0";
int baudrate = 115200;
double maxYawRate = 80.0;
double maxSpeed = 1.0;

double joyTime = 0;
float joyFwd = 0;
float joyLeft = 0;
float joyYaw = 0;

bool serialOpen = false;

void joystickHandler(const sensor_msgs::msg::Joy::ConstSharedPtr joy)
{
  joyFwd = joy->axes[4];
  joyLeft = joy->axes[3];
  joyYaw = joy->axes[0];
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto nh = rclcpp::Node::make_shared("teleopJoyController");

  nh->declare_parameter<string>("serialPort", serialPort);
  nh->declare_parameter<int>("baudrate", baudrate);
  nh->declare_parameter<double>("maxYawRate", maxYawRate);
  nh->declare_parameter<double>("maxSpeed", maxSpeed);

  nh->get_parameter("serialPort", serialPort);
  nh->get_parameter("baudrate", baudrate);
  nh->get_parameter("maxYawRate", maxYawRate);
  nh->get_parameter("maxSpeed", maxSpeed);

  auto subJoystick = nh->create_subscription<sensor_msgs::msg::Joy>("/joy", 5, joystickHandler);

  auto pubSpeed = nh->create_publisher<geometry_msgs::msg::TwistStamped>("/cmd_vel", 5);
  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header.frame_id = "vehicle";

  serial::Serial *motorCtrSerial = new serial::Serial();
  serial::Timeout timeout(serial::Timeout::simpleTimeout(500));
  motorCtrSerial->setTimeout(timeout);
  motorCtrSerial->setPort(serialPort.c_str());
  motorCtrSerial->setBaudrate(baudrate);

  float value;
  size_t size = sizeof(value);
  uint8_t *serialBuffer = static_cast<uint8_t*>(alloca(3 * size + 1));

  int initFrameCount = 0;
  rclcpp::Rate rate(50);
  bool status = rclcpp::ok();
  while (status) {
    rclcpp::spin_some(nh);

    cmd_vel.header.stamp = nh->now();
    cmd_vel.twist.linear.x = maxSpeed * joyFwd;
    cmd_vel.twist.linear.y = maxSpeed / 2.0 * joyLeft;
    cmd_vel.twist.angular.z = maxYawRate * PI / 180.0 * joyYaw;

    pubSpeed->publish(cmd_vel);

    if (serialOpen) {
      value = cmd_vel.twist.linear.x;
      memcpy(serialBuffer, &value, size);
      value = cmd_vel.twist.linear.y;
      memcpy(serialBuffer + size, &value, size);
      value = cmd_vel.twist.angular.z;
      memcpy(serialBuffer + 2 * size, &value, size);
      serialBuffer[3 * size] = '\n';

      motorCtrSerial->write(serialBuffer, 3 * size + 1);
    } else {
      if (initFrameCount % 50 == 0) {
        try {
          motorCtrSerial->open();
        } catch (serial::IOException) {
        }

        if (motorCtrSerial->isOpen()) {
          serialOpen = true;
          RCLCPP_INFO(nh->get_logger(), "Serial port open.");
        } else {
          RCLCPP_INFO(nh->get_logger(), "Opening serial port %s...", serialPort.c_str());
        }
      }
      initFrameCount++;
    }

    status = rclcpp::ok();
    rate.sleep();
  }

  if (serialOpen) motorCtrSerial->close();

  return 0;
}
