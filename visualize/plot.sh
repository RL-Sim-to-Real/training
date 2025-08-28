#!/usr/bin/env bash


# Average reward per time step
python tb_plot.py   --logdir ../logs --title "Average Reward" --yaxis "Average Reward"  --out figs_pub/img_prioception   --tags "episode/sum_reward"   --groups "\
Img+Prioception (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
Img+Prioception (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
Img+Prioception (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
Img+Prioception (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"   --smoothing 0.9


python tb_plot.py   --logdir ../logs --title "Average Reward" --yaxis "Average Reward"  --out figs_pub/img   --tags "episode/sum_reward"   --groups "\
Img (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-2*;
Img (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-2*;
Img (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-2*;
Img (Torque)=**/PandaPickCubeCartesianModified-joint-torque-2-*"   --smoothing 0.9

# Success rate
python tb_plot.py   --logdir ../logs --title "Success Rate" --yaxis "Success"  --out figs_pub/img_prioception   --tags "episode/success"   --groups "\
Img+Prioception (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
Img+Prioception (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
Img+Prioception (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
Img+Prioception (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"   --smoothing 0.9

python tb_plot.py   --logdir ../logs --title "Success Rate" --yaxis "Success"  --out figs_pub/img   --tags "episode/success"   --groups "\
Img (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-2*;
Img (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-2*;
Img (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-2*;
Img (Torque)=**/PandaPickCubeCartesianModified-joint-torque-2*"   --smoothing 0.9


# Cube Collision
python tb_plot.py   --logdir ../logs --title "Cube Collision" --yaxis "Collision"  --out figs_pub/img_prioception   --tags "episode/cube_collision"   --groups "\
Img+Prioception (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
Img+Prioception (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
Img+Prioception (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
Img+Prioception (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"   --smoothing 0.9


python tb_plot.py   --logdir ../logs --title "Cube Collision" --yaxis "Collision"  --out figs_pub/img   --tags "episode/cube_collision"   --groups "\
Img (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-2*;
Img (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-2*;
Img (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-2*;
Img (Torque)=**/PandaPickCubeCartesianModified-joint-torque-2*"   --smoothing 0.9


# Floor Collision
python tb_plot.py   --logdir ../logs --title "Floor Collision" --yaxis "Collision"  --out figs_pub/img_prioception   --tags "episode/floor_collision"   --groups "\
Img+Prioception (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
Img+Prioception (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
Img+Prioception (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
Img+Prioception (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"   --smoothing 0.9


python tb_plot.py   --logdir ../logs --title "Floor Collision" --yaxis "Collision"  --out figs_pub/img   --tags "episode/floor_collision"   --groups "\
Img (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-2*;
Img (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-2*;
Img (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-2*;
Img (Torque)=**/PandaPickCubeCartesianModified-joint-torque-2*"   --smoothing 0.9

