"""Seeded per-world dynamics and appearance randomization."""
import jax
import jax.numpy as jp
from mujoco.mjx._src import math


def domain_randomize(m, *, model, num_worlds, seed, vision):
    box = model.geom('box').id
    floor = model.geom('floor').id

    def sample(key):
        gain_key, color_key, camera_key, angle_key, light_key, shadow_key = jax.random.split(key, 6)
        gain = jax.random.uniform(gain_key, (), minval=0.9, maxval=1.1)
        fields = {
            'actuator_gainprm': m.actuator_gainprm.at[:, 0].multiply(gain),
            'actuator_biasprm': m.actuator_biasprm.at[:, 1:3].multiply(gain),
        }
        if vision:
            # MuJoCo uses -1 for untextured geoms, not Madrona's -2 sentinel.
            fields['geom_rgba'] = m.geom_rgba.at[box].set(jp.array([
                jax.random.uniform(color_key, (), minval=0.7, maxval=1.), 0., 0., 1.]))
            fields['geom_matid'] = m.geom_matid.at[box].set(-1).at[floor].set(-1)
            fields['cam_pos'] = m.cam_pos + jax.random.uniform(
                camera_key, m.cam_pos.shape, minval=-0.01, maxval=0.01)
            angle = jax.random.uniform(angle_key, (), minval=-0.2, maxval=0.2) * jp.pi / 180
            rotation = math.axis_angle_to_quat(jp.array([0., 1., 0.]), angle)
            fields['cam_quat'] = jax.vmap(lambda q: math.quat_mul(rotation, q))(m.cam_quat)
            pos = jax.random.uniform(light_key, m.light_pos.shape,
                                     minval=jp.array([1.5, -0.2, 0.8]),
                                     maxval=jp.array([3., 0.2, 1.5]))
            direction = jp.array([0.661, -0.001, 0.179]) - pos
            fields['light_pos'] = pos
            fields['light_dir'] = direction / jp.linalg.norm(direction, axis=-1, keepdims=True)
            fields['light_castshadow'] = jax.random.bernoulli(
                shadow_key, 0.75, m.light_castshadow.shape).astype(m.light_castshadow.dtype)
        return fields

    fields = jax.vmap(sample)(jax.random.split(jax.random.PRNGKey(seed), num_worlds))
    axes = jax.tree.map(lambda _: None, m).tree_replace({k: 0 for k in fields})
    return m.tree_replace(fields), axes
