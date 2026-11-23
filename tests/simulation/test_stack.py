"""Regression checks for the original tasks; no robot hardware imports."""
import numpy as np
import pytest
from training_cli import parse_args


def test_cli_preserves_benchmark_flags():
    a = parse_args(argv=['--num_envs=8', '--batch_size=8', '--vision',
                         '--proprioception', '--actuator=velocity', '--seed=7'])
    assert a.num_envs == 8 and a.seed == 7 and a.proprioception
    assert a.domain_randomization


@pytest.mark.parametrize('argv', [
    ['--num_envs=0'], ['--num_envs=3'], ['--vision', '--impl=jax'],
    ['--play_only'], ['--action=joint', '--actuator=position'],
    ['--action=joint_increment', '--actuator=velocity']])
def test_invalid_cli(argv):
    with pytest.raises(SystemExit):
        parse_args(argv=argv)


@pytest.mark.parametrize('task', ['PandaPickCuboid', 'PandaPushCuboid'])
def test_randomization_seed_and_batch(task):
    import jax
    from simulation import make_env
    from randomization import domain_randomize
    args = parse_args(task, ['--smoke', '--impl=jax', '--no-vision'])
    env, _ = make_env(args)
    m = env.mjx_model
    kw = dict(model=env.mj_model, num_worlds=4, vision=False)
    a, axes = domain_randomize(m, seed=42, **kw)
    b, _ = domain_randomize(m, seed=42, **kw)
    c, _ = domain_randomize(m, seed=43, **kw)
    np.testing.assert_array_equal(a.actuator_gainprm, b.actuator_gainprm)
    assert not np.array_equal(a.actuator_gainprm, c.actuator_gainprm)
    assert not np.array_equal(a.actuator_gainprm[0], a.actuator_gainprm[1])
    assert axes.actuator_gainprm == 0
    assert a.actuator_gainprm.shape[0] == 4
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(a))


@pytest.mark.integration
@pytest.mark.parametrize('task', ['PandaPickCuboid', 'PandaPushCuboid'])
@pytest.mark.parametrize('actuator', ['position', 'velocity'])
def test_parallel_reset_step_autoreset(task, actuator):
    import jax
    import jax.numpy as jp
    from simulation import make_env
    args = parse_args(task, ['--smoke', '--impl=jax', '--no-vision', f'--actuator={actuator}'])
    env, cfg = make_env(args)
    original_model = env.unwrapped.mjx_model
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    keys = jax.random.split(jax.random.PRNGKey(0), args.num_envs)
    state = reset(keys)
    initial_obs = np.asarray(state.obs).copy()
    shape = state.obs.shape
    for _ in range(cfg.episode_length + 1):
        state = step(state, jp.zeros((args.num_envs, env.action_size)))
        assert state.obs.shape == shape
        assert np.isfinite(state.obs).all()
        assert np.isfinite(state.data.qpos).all()
        assert np.isfinite(state.reward).all()
    # Model temporarily bound during vmap must not retain a leaked tracer.
    assert env.unwrapped.mjx_model is original_model
    repeat = reset(keys)
    np.testing.assert_allclose(repeat.obs, initial_obs, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize('task', ['PandaPickCuboid', 'PandaPushCuboid'])
@pytest.mark.parametrize('randomize', [False, True])
def test_warp_vision(task, randomize):
    import jax
    import jax.numpy as jp
    from simulation import make_env
    # Explicit GPU validation must fail, not silently skip, on a CPU-only host.
    assert jax.default_backend() == 'gpu', 'GPU validation requires a CUDA JAX device'
    flags = ['--smoke', '--vision', '--proprioception']
    if not randomize:
        flags.append('--no-domain_randomization')
    args = parse_args(task, flags)
    env, _ = make_env(args)
    state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(0), 2))
    step = jax.jit(env.step)
    for _ in range(5):
        state = step(state, jp.zeros((2, env.action_size)))
        pixels = np.asarray(state.obs['pixels/view_0'])
        assert pixels.shape == (2, 64, 64, 3)
        assert np.isfinite(pixels).all() and pixels.std() > 0
        red = (pixels[..., 0] > pixels[..., 1] * 1.5 + .05) & (pixels[..., 0] > pixels[..., 2] * 1.5 + .05)
        assert red.sum() > 3, 'The rendered cuboid must be visible, not just a nonuniform background'
        assert np.isfinite(state.obs['_prop']).all()
        assert np.isfinite(state.reward).all()


def test_packed_renderer_channels(monkeypatch):
    import jax.numpy as jp
    from mujoco_playground._src.modern_backend import WarpRenderer
    from mujoco import mjx
    from types import SimpleNamespace
    renderer = WarpRenderer.__new__(WarpRenderer)
    renderer.width = renderer.height = renderer.ncam = 1
    renderer.context = SimpleNamespace(pytree=lambda: None)
    token = object()
    monkeypatch.setattr(mjx, 'refit_bvh', lambda model, data, ctx: data)
    monkeypatch.setattr(mjx, 'render', lambda *args: (jp.array([0xFF123456], dtype=jp.uint32), None, token))
    rgb, data = renderer.render(None, None)
    np.testing.assert_array_equal(rgb, [[[[0x12, 0x34, 0x56]]]])
    assert data is token


def test_contact_sensors_match_native_contacts():
    import mujoco
    from mujoco_playground import manipulation
    env = manipulation.load('PandaPickCuboid', config_overrides={'vision': False})
    m = env.mj_model
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe('low_home').id)
    # Move box through the hand, testing both collision and separation.
    seen_collision = False
    for offset in np.linspace(-.08, .08, 9):
        mujoco.mj_forward(m, d)
        d.qpos[env._obj_qposadr:env._obj_qposadr+3] = d.geom_xpos[env._left_finger_geom] + [0, 0, offset]
        mujoco.mj_forward(m, d)
        for pair, adr in env._contact_sensors.items():
            distances = [c.dist for c in d.contact if tuple(sorted(c.geom)) == pair]
            expected = bool(distances and min(distances) < 0)
            seen_collision |= expected
            assert bool(d.sensordata[adr] < 0) == expected
    assert seen_collision


def test_explicit_values_override_smoke_defaults():
    args = parse_args(argv=['--smoke', '--num_envs=4', '--batch_size=4',
                            '--rollout_steps=8', '--episode_length=6'])
    assert (args.num_envs, args.batch_size, args.rollout_steps, args.episode_length) == (4, 4, 8, 6)
