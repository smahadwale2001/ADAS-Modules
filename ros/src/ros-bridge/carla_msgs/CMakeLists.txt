cmake_minimum_required(VERSION 2.8.3)
project(carla_msgs)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
)

add_message_files(
  DIRECTORY msg
  FILES
  CarlaEgoVehicleControl.msg
  CarlaEgoVehicleStatus.msg
  CarlaEgoVehicleInfoWheel.msg
  CarlaEgoVehicleInfo.msg
  CarlaCollisionEvent.msg
  CarlaLaneInvasionEvent.msg
  CarlaWorldInfo.msg
  CarlaActorInfo.msg
  CarlaActorList.msg
  CarlaControl.msg
  CarlaStatus.msg
  CarlaWalkerControl.msg
  error.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)


catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
  geometry_msgs
)
