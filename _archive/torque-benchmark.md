# Historical torque benchmark

The removed shell launcher used PandaPickCuboid, 50,000,000 timesteps, seeds
0–9, GPU 0, torque/joint actions at scale 1.0, vision plus proprioception,
TensorBoard and training metrics, and a five-second pause between seeds.

This experiment predates the supported position/velocity stack.
`task archive:benchmark-prop-torque` reports that limitation instead of launching
an unsupported run. The original script remains available in Git history.
