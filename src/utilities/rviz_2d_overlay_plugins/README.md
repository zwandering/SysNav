# rviz_2d_overlay_plugins

Plugins and message definitions for displaying 2D overlays over the
RViz2 3D scene.

Based on the [jsk_visualization](https://github.com/jsk-ros-pkg/jsk_visualization)
package, which is currently only released for ROS1, under the 3-Clause BSD license.

See the [hosted doxygen documentation](https://docs.ros.org/en/rolling/p/rviz_2d_overlay_plugins/generated/doxygen) for
both usage information and source code documentation.

![Screenshot showing the robot velocity as an overlay above the RViz 3D Scene, as well as the expanded properties of the plugin](rviz_2d_overlay_plugins/doc/screenshot_vel_overlay.png)

![Screenshot showing the PieChartDisplay, a circular gauge](rviz_2d_overlay_plugins/doc/screenshot_PieChartDisplay.png)


```
colcon build --packages-select rviz_2d_overlay_msgs rviz_2d_overlay_plugins
```