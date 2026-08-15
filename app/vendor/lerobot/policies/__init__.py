# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from lerobot.utils.action_interpolator import ActionInterpolator as ActionInterpolator

from .factory import get_policy_class, make_policy, make_policy_config, make_pre_post_processors
from .pi05.configuration_pi05 import PI05Config as PI05Config
from .pretrained import PreTrainedPolicy as PreTrainedPolicy
from .utils import make_robot_action, prepare_observation_for_inference

# NOTE: Policy modeling classes (e.g., GaussianActorPolicy) are intentionally NOT re-exported here.
# They have heavy optional dependencies and are loaded lazily via get_policy_class().
# Import directly: ``from lerobot.policies.gaussian_actor.modeling_gaussian_actor import GaussianActorPolicy``

__all__ = [
    # Configuration classes
    "PI05Config",
    # Base class
    "PreTrainedPolicy",
    # RTC utilities
    "ActionInterpolator",
    # Utility functions
    "make_robot_action",
    "prepare_observation_for_inference",
    # Factory functions
    "get_policy_class",
    "make_policy",
    "make_policy_config",
    "make_pre_post_processors",
]
