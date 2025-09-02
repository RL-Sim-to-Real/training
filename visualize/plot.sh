#!/usr/bin/env bash


# Average reward per time step
python tb_plot.py   --logdir ../logs-aug27 --title "Average Reward" --yaxis "Average Reward"  --out figs_pub/img_prioception   --tags "episode/sum_reward"   --groups "\
Img+Prioception (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
Img+Prioception (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
Img+Prioception (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
Img+Prioception (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"   --bin_count 200


python tb_plot.py   --logdir ../logs-aug27 --title "Average Reward" --yaxis "Average Reward"  --out figs_pub/img   --tags "episode/sum_reward"   --groups "\
Img (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-2*;
Img (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-2*;
Img (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-2*;
Img (Torque)=**/PandaPickCubeCartesianModified-joint-torque-2*"   --bin_count 200

# Success rate
python tb_plot.py   --logdir ../logs-aug27 --title "Success Rate" --yaxis "Success"  --out figs_pub/img_prioception   --tags "episode/success"   --groups "\
Img+Prioception (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
Img+Prioception (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
Img+Prioception (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
Img+Prioception (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"   --bin_count 200

python tb_plot.py   --logdir ../logs-aug27 --title "Success Rate" --yaxis "Success"  --out figs_pub/img   --tags "episode/success"   --groups "\
Img (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-2*;
Img (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-2*;
Img (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-2*;
Img (Torque)=**/PandaPickCubeCartesianModified-joint-torque-2*"   --bin_count 200


# Cube Collision
python tb_plot.py   --logdir ../logs-aug27 --title "Cube Collision" --yaxis "Collision"  --out figs_pub/img_prioception   --tags "episode/cube_collision"   --groups "\
Img+Prioception (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
Img+Prioception (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
Img+Prioception (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
Img+Prioception (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"   --bin_count 200


python tb_plot.py   --logdir ../logs-aug27 --title "Cube Collision" --yaxis "Collision"  --out figs_pub/img   --tags "episode/cube_collision"   --groups "\
Img (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-2*;
Img (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-2*;
Img (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-2*;
Img (Torque)=**/PandaPickCubeCartesianModified-joint-torque-2*"   --bin_count 200


# Floor Collision
python tb_plot.py   --logdir ../logs-aug27 --title "Floor Collision" --yaxis "Collision"  --out figs_pub/img_prioception   --tags "episode/floor_collision"   --groups "\
Img+Prioception (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
Img+Prioception (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
Img+Prioception (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
Img+Prioception (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"   --bin_count 200


python tb_plot.py   --logdir ../logs-aug27 --title "Floor Collision" --yaxis "Collision"  --out figs_pub/img   --tags "episode/floor_collision"   --groups "\
Img (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-2*;
Img (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-2*;
Img (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-2*;
Img (Torque)=**/PandaPickCubeCartesianModified-joint-torque-2*"   --bin_count 200

