"""Shared environment construction for training and validation."""
import functools


def make_env(args, *, num_envs=None, seed=None):
    import jax
    from mujoco_playground import manipulation, wrapper
    from randomization import domain_randomize

    num_envs = num_envs or args.num_envs
    cfg = manipulation.get_default_config(args.env_name)
    if 'impl' not in cfg:
        raise RuntimeError('Apply patches/playground-modern.patch using task setup first.')
    cfg.impl = args.impl
    cfg.vision = args.vision
    cfg.proprioception = args.proprioception
    cfg.actuator = args.actuator
    cfg.action = args.action
    cfg.action_scale = args.action_scale
    cfg.episode_length = args.episode_length or int(
        (6 if args.env_name == 'PandaPickCuboid' else 12) / cfg.ctrl_dt)
    cfg.box_init_range = 0.1 if args.env_name == 'PandaPickCuboid' else 0.03
    cfg.success_threshold = 0.03 if args.env_name == 'PandaPickCuboid' else 0.01
    cfg.action_history_length = 5
    cfg.obs_noise.brightness = [0.75, 2.0] if args.domain_randomization else [1.0, 1.0]
    cfg.vision_config.render_batch_size = num_envs
    cfg.vision_config.render_width = args.render_size
    cfg.vision_config.render_height = args.render_size
    if args.vision and args.impl != 'warp':
        raise ValueError('Vision requires --impl=warp.')
    if args.impl == 'warp' and jax.default_backend() != 'gpu':
        raise RuntimeError('MJX-Warp requires a working NVIDIA driver and JAX CUDA installation.')
    env = manipulation.load(args.env_name, config=cfg)
    randomizer = None
    if args.domain_randomization:
        randomizer = functools.partial(
            domain_randomize, model=env.mj_model, num_worlds=num_envs,
            seed=args.seed if seed is None else seed, vision=args.vision)
    # Vision now uses native MJX rendering: no Madrona-specific tiling/wrapper.
    wrapped = wrapper.wrap_for_brax_training(
        env, episode_length=cfg.episode_length, action_repeat=args.action_repeat,
        randomization_fn=randomizer)
    return wrapped, cfg
