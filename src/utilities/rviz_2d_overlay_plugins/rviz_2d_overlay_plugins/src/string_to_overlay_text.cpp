// publishes rviz_2d_overlay_msgs/msg/OverlayText from std_msgs/msg/String

// test with:
// ros2 topic pub /chatter std_msgs/String "data: Hello world"
// ros2 run rviz_2d_overlay_plugins string_to_overlay_text
// ros2 run rviz_2d_overlay_plugins string_to_overlay_text --ros-args -p string_topic:=chatter -p fg_color:=r
// ros2 launch rviz_2d_overlay_plugins string_to_overlay_text_example.launch.py

#include <iostream>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "rviz_2d_overlay_msgs/msg/overlay_text.hpp"
#include "std_msgs/msg/string.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"



using namespace std::chrono_literals;
using std::placeholders::_1;

class Rviz2dString : public rclcpp::Node
{
    rcl_interfaces::msg::SetParametersResult parametersCallback(const std::vector<rclcpp::Parameter> &parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";
        for (const auto &param: parameters)
        {
            RCLCPP_INFO_STREAM(this->get_logger(), "Param update: " << param.get_name().c_str() << ": " <<param.value_to_string().c_str());
            if (param.get_name() == "string_topic")
            {
                string_topic = param.as_string();
                overlay_text_topic = string_topic + "_overlay_text";
            }
            if (param.get_name() == "fg_color")
            {
                fg_color = param.as_string();
            }            
        }
        RCLCPP_INFO_STREAM(this->get_logger(), "subscribed: " << string_topic << " publishing: " << overlay_text_topic << " fg color: " << fg_color << " ");
        return result;
    }

public:
    Rviz2dString() : Node("rviz2d_from_string_node")
    {      
        this->declare_parameter<std::string>("string_topic", "");
        this->declare_parameter<std::string>("fg_color", "");
        this->get_parameter("string_topic", string_topic);
        this->get_parameter("fg_color", fg_color);
        if (string_topic == "")
        {
            RCLCPP_ERROR_STREAM(this->get_logger(), "Parameter string_topic not set");
            string_topic = "/chatter";   
        }
        overlay_text_topic = string_topic + "_overlay_text";

        sub_st_ = this->create_subscription<std_msgs::msg::String>(string_topic, 10, std::bind(&Rviz2dString::strCallback, this, _1));
        pub_ov_ = this->create_publisher<rviz_2d_overlay_msgs::msg::OverlayText>(overlay_text_topic, 1);
        callback_handle_ = this->add_on_set_parameters_callback(std::bind(&Rviz2dString::parametersCallback, this, std::placeholders::_1));
        RCLCPP_INFO_STREAM(this->get_logger(), "Node started: " << this->get_name() << " subscribed: " << string_topic << " publishing: " << overlay_text_topic << " ");
    }

private:
    // Callback for string
    void strCallback(const std_msgs::msg::String &st_msg)
    {
        rviz_2d_overlay_msgs::msg::OverlayText ov_msg;
        ov_msg.text = st_msg.data;
        ov_msg.fg_color.a = 1.0;

        // https://github.com/jkk-research/colors
        if (fg_color == "r") // red
        {
            ov_msg.fg_color.r = 0.96f;
            ov_msg.fg_color.g = 0.22f;
            ov_msg.fg_color.b = 0.06f;
        }
        else if (fg_color == "g") // green
        {
            ov_msg.fg_color.r = 0.30f;
            ov_msg.fg_color.g = 0.69f;
            ov_msg.fg_color.b = 0.31f;
        }
        else if (fg_color == "b") // blue
        {
            ov_msg.fg_color.r = 0.02f;
            ov_msg.fg_color.g = 0.50f;
            ov_msg.fg_color.b = 0.70f;
        }
        else if (fg_color == "k") // black
        {
            ov_msg.fg_color.r = 0.19f;
            ov_msg.fg_color.g = 0.19f;
            ov_msg.fg_color.b = 0.23f;
        }
        else if (fg_color == "w") // white
        {
            ov_msg.fg_color.r = 0.89f;
            ov_msg.fg_color.g = 0.89f;
            ov_msg.fg_color.b = 0.93f;
        }    
        else if (fg_color == "p") // pink
        {
            ov_msg.fg_color.r = 0.91f;
            ov_msg.fg_color.g = 0.12f;
            ov_msg.fg_color.b = 0.39f;
        }                     
        else
        { // yellow
            ov_msg.fg_color.r = 0.94f;
            ov_msg.fg_color.g = 0.83f;
            ov_msg.fg_color.b = 0.07f;
        }        
        ov_msg.width = 200;
        ov_msg.height = 40;
        pub_ov_->publish(ov_msg);
    }

    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_st_;
    std::string string_topic, overlay_text_topic, fg_color;
    rclcpp::Publisher<rviz_2d_overlay_msgs::msg::OverlayText>::SharedPtr pub_ov_;
    OnSetParametersCallbackHandle::SharedPtr callback_handle_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Rviz2dString>());
    rclcpp::shutdown();
    return 0;
}