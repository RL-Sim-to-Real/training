#!/usr/bin/env bash

# python tb_plot.py   --logdir ../logs   --out figs_pub/cartesian_position   --tags "episode/sum_reward"   --groups "\
# Img=**/PandaPickCubeCartesianModified-cartesian_increment-position-20*;\
# Img+Prioception=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*"   --smoothing 0.9

# python tb_plot.py   --logdir ../logs   --out figs_pub/velocity   --tags "episode/sum_reward"   --groups "\
# Img=**/PandaPickCubeCartesianModified-joint_increment-velocity-20*;\
# Img+Prioception=**/PandaPickCubeCartesianModified-joint_increment-velocity-_prop-*"   --smoothing 0.9

# python tb_plot.py   --logdir ../logs   --out figs_pub/position   --tags "episode/sum_reward"   --groups "\
# # Img=**/PandaPickCubeCartesianModified-joint_increment-position-20*;\
# Img+Prioception=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*"   --smoothing 0.9

# python tb_plot.py   --logdir ../logs   --out figs_pub/torque   --tags "episode/sum_reward"   --groups "\
# CartPos=**/PandaPickCubeCartesianModified-joint_increment-torque-20*;\
# CartPos+Img=**/PandaPickCubeCartesianModified-joint_increment-torque-_prop-*"   --smoothing 0.9

python tb_plot.py   --logdir ../logs --title "Average Reward" --yaxis "Average Reward"  --out figs_pub/img_prioception   --tags "episode/sum_reward"   --groups "\
Img+Prioception (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
Img+Prioception (Velocity Increment)=**/PandaPickCubeCartesianModified-joint_increment-velocity-_prop-*;
Img+Prioception (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
Img+Prioception (Torque Increment)=**/PandaPickCubeCartesianModified-joint_increment-torque-_prop-*"   --smoothing 0.9

python tb_plot.py   --logdir ../logs --title "Success Rate" --yaxis "Success"  --out figs_pub/img_prioception   --tags "episode/reward/success"   --groups "\
Img+Prioception (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
Img+Prioception (Velocity Increment)=**/PandaPickCubeCartesianModified-joint_increment-velocity-_prop-*;
Img+Prioception (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
Img+Prioception (Torque Increment)=**/PandaPickCubeCartesianModified-joint_increment-torque-_prop-*"   --smoothing 0.9

