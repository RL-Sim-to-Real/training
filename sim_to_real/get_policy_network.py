# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics




def make_inference_fn(
    normalize_observations: bool = False,
    action_size: int = 4,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    include_prop: bool = False,
):

  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize
  obs_sizes = {'pixels/view_0': (64, 64, 3)}
  if not include_prop:
    ppo_network = network_factory(
        obs_sizes,
        action_size,
        preprocess_observations_fn=normalize,
    )
  else:
    obs_sizes['_prop'], obs_key = (20,), '_prop'
    ppo_network = network_factory(
        obs_sizes,
        action_size,
        preprocess_observations_fn=normalize,
        policy_obs_key=obs_key,
        value_obs_key=obs_key,
    )
  make_policy = ppo_networks.make_inference_fn(ppo_network)
  return make_policy