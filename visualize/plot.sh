#!/usr/bin/env bash


# Episodic Return
# python tb_plot2.py   --logdir ../logs --title "Episodic Return" --yaxis "Return"  --out figs_pub/img_prioception   --tags "episode/sum_reward"   --groups "\
# Img+Prop (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
# Img+Prop (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
# Img+Prop (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
# Img+Prop (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"\
#    --bin_count 200 --verbose --suffix "_prioception" 


python tb_plot_individual.py   --logdir ../logs --title "Episodic Return" --yaxis "Return"  --out figs_pub/img   --tags "episode/sum_reward"   --groups "\
Cartesian Increment=**/PandaPickCubeCartesianModified-cartesian_increment-position-seed*;
Velocity=**/PandaPickCubeCartesianModified-joint-velocity-seed*;
Position Increment=**/PandaPickCubeCartesianModified-joint_increment-position-seed*;
Torque=**/PandaPickCubeCartesianModified-joint-torque-seed*"   --bin_count 200

# Success rate
# python tb_plot2.py   --logdir ../logs --title "Success Rate" --yaxis "Success"  --out figs_pub/img_prioception   --tags "episode/success"   --groups "\
# Img+Prop (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
# Img+Prop (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
# Img+Prop (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
# Img+Prop (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"\
#    --bin_count 200 --verbose --suffix "_prioception"

python tb_plot_individual.py   --logdir ../logs --title "Success Rate" --yaxis "Success"  --out figs_pub/img   --tags "episode/success"   --groups "\
Cartesian Increment=**/PandaPickCubeCartesianModified-cartesian_increment-position-seed*;
Velocity=**/PandaPickCubeCartesianModified-joint-velocity-seed*;
Position Increment=**/PandaPickCubeCartesianModified-joint_increment-position-seed*;
Torque=**/PandaPickCubeCartesianModified-joint-torque-seed*"   --bin_count 200


# Cube Collision
# python tb_plot2.py   --logdir ../logs --title "Episodic Cube Collision" --yaxis "Collision"  --out figs_pub/img_prioception   --tags "episode/cube_collision"   --groups "\
# Img+Prop (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
# Img+Prop (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
# Img+Prop (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
# Img+Prop (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"\
#    --bin_count 200 --verbose --suffix "_prioception"


python tb_plot_individual.py   --logdir ../logs --title "Episodic Cube Collision" --yaxis "Collision"  --out figs_pub/img   --tags "episode/cube_collision"   --groups "\
Cartesian Increment=**/PandaPickCubeCartesianModified-cartesian_increment-position-seed*;
Velocity=**/PandaPickCubeCartesianModified-joint-velocity-seed*;
Position Increment=**/PandaPickCubeCartesianModified-joint_increment-position-seed*;
Torque=**/PandaPickCubeCartesianModified-joint-torque-seed*"   --bin_count 200


# Floor Collision
# python tb_plot2.py   --logdir ../logs --title "Episodic Floor Collision" --yaxis "Collision"  --out figs_pub/img_prioception   --tags "episode/floor_collision"   --groups "\
# Img+Prop (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
# Img+Prop (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
# Img+Prop (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
# Img+Prop (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"\
#    --bin_count 200 --verbose --suffix "_prioception" 


python tb_plot_individual.py   --logdir ../logs --title "Episodic Floor Collision" --yaxis "Collision"  --out figs_pub/img   --tags "episode/floor_collision"   --groups "\
Cartesian Increment=**/PandaPickCubeCartesianModified-cartesian_increment-position-seed*;
Velocity=**/PandaPickCubeCartesianModified-joint-velocity-seed*;
Position Increment=**/PandaPickCubeCartesianModified-joint_increment-position-seed*;
Torque=**/PandaPickCubeCartesianModified-joint-torque-seed*"   --bin_count 200


# Jerk
# python tb_plot2.py   --logdir ../logs --title "Episodic Jerk" --yaxis "Jerk"  --out figs_pub/img_prioception   --tags "episode/jerk"   --groups "\
# Img+Prop (Cartesian Increment)=**/PandaPickCubeCartesianModified-cartesian_increment-position-_prop-*;
# Img+Prop (Velocity)=**/PandaPickCubeCartesianModified-joint-velocity-_prop-*;
# Img+Prop (Position Increment)=**/PandaPickCubeCartesianModified-joint_increment-position-_prop-*;
# Img+Prop (Torque)=**/PandaPickCubeCartesianModified-joint-torque-_prop-*"\
#    --bin_count 200 --verbose --suffix "_prioception"


python tb_plot_individual.py   --logdir ../logs --title "Episodic Jerk" --yaxis "Jerk"  --out figs_pub/img   --tags "episode/jerk"   --groups "\
Cartesian Increment=**/PandaPickCubeCartesianModified-cartesian_increment-position-seed*;
Velocity=**/PandaPickCubeCartesianModified-joint-velocity-seed*;
Position Increment=**/PandaPickCubeCartesianModified-joint_increment-position-seed*;
Torque=**/PandaPickCubeCartesianModified-joint-torque-seed*"   --bin_count 200

