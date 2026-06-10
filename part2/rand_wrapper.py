"""
rand_wrapper.py
===============
Gymnasium wrapper that randomizes the cube mass in PandaPush-v3.

Modes
-----
none  : no randomization — mass is never changed.
udr   : Uniform Domain Randomization — mass sampled uniformly from
        [mass_range[0], mass_range[1]] at every episode reset.
adr   : Automatic Domain Randomization — maintains a centre and a
        half-width (delta).  After every `adr_buffer_size` episodes the
        half-width expands if the rolling success rate is above the
        threshold, otherwise it shrinks.  The centre also shifts using a
        performance-gradient rule so the distribution can drift to cover
        the target domain over time.

Design notes
------------
* Success is tracked across the *whole* episode, not just the final step,
  because panda-gym does not guarantee is_success is present on every
  terminal info dict.
* ADR update is called at the top of reset() (not in step()) so we always
  have the complete episode outcome before starting the next one.
* The mass is injected BEFORE super().reset() so the first observation of
  every episode already reflects the new physics.
* ADR history (episode, center, delta) is stored in self.adr_history for
  saving to CSV in train_sb3.py after training.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym


class RandomizationWrapper(gym.Wrapper):

    def __init__(self,env,mass_range=(1.0, 1.0),mode="none",adr_init_mass=1.0,adr_success_threshold=0.70,adr_expand_step=0.15,adr_shrink_step=0.05,adr_buffer_size=100,adr_center_lr=0.1,):
        super().__init__(env)

        self.mode = mode
        self.mass_range = mass_range


        # Hard limits — sampled mass will never leave this interval
        self.mass_min_limit, self.mass_max_limit = mass_range

        # UDR: effective range equals the global limits
        self.mass_min = self.mass_min_limit
        self.mass_max = self.mass_max_limit

        # ADR state
        self.adr_center = float(adr_init_mass)
        self.adr_delta = 0.10   # initial half-width (kg)
        self.adr_success_threshold = float(adr_success_threshold)
        self.adr_expand_step = float(adr_expand_step)
        self.adr_shrink_step = float(adr_shrink_step)
        self.adr_buffer_size = int(adr_buffer_size)
        self.adr_center_lr = float(adr_center_lr)
        self._adr_success_buffer: list[float] = []

        # Per-episode success flag — robust to missing final info key
        # Robust to panda-gym not guaranteeing is_success on every terminal step
        self._current_ep_success = False

        # History for analysis_plots.py ADR evolution curve
        self.adr_history = []   # list of (episode_idx, center, delta)

        # Mass samples for UDR histogram plot
        self.mass_samples = []

        self._episode_count = 0
        self.last_mass = None
        self.last_sample_type = "fixed"

    # ADR boundary properties
    @property
    def adr_mass_low(self):
        return float(np.clip(
            self.adr_center - self.adr_delta,
            self.mass_min_limit, self.mass_max_limit,
        ))

    @property
    def adr_mass_high(self):
        """Upper bound of the current ADR sampling interval."""
        return float(np.clip(
            self.adr_center + self.adr_delta,
            self.mass_min_limit, self.mass_max_limit,
        ))

    # Mass Sampling
    def _sample_mass(self):
        """
        Sample a new cube mass according to the current mode.

        Returns None for mode='none' (no randomization).
        For UDR: uniform draw from [mass_min_limit, mass_max_limit].
        For ADR: uniform draw from the current adaptive interval.
        """
        if self.mode == "none":
            return None

        elif self.mode == "udr":
            # Uniform Domain Randomization:
            # sample uniformly over the full configured mass range
            self.last_sample_type = "uniform"
            self.mass_min = self.mass_min_limit
            self.mass_max = self.mass_max_limit
            mass = float(np.random.uniform(self.mass_min_limit, self.mass_max_limit))
            self.mass_samples.append(mass)
            return mass

        elif self.mode == "adr":
            # Automatic Domain Randomization:
            # sample uniformly over the current adaptive interval
            self.last_sample_type = "adr"
            self.mass_min = self.adr_mass_low
            self.mass_max = self.adr_mass_high
            mass = float(np.random.uniform(self.mass_min, self.mass_max))
            self.mass_samples.append(mass)
            return mass

        else:
            raise NotImplementedError(f"Sampling strategy '{self.mode}' is not implemented yet.")

    # ADR range update

    def _adr_update_range(self, is_success):
        """
        Append episode outcome to the buffer.  When the buffer is full,
        update the ADR centre and delta.

        Centre update (performance-gradient rule, paper-aligned):
            center += lr * (rate - threshold)
        This drives the centre toward the difficulty boundary and allows
        the distribution to migrate toward the target domain over time.

        Dynamic padding prevents the centre from pinning against the hard
        limits, which would collapse the sampling window to zero width.
        """
        self._adr_success_buffer.append(float(is_success))

        if len(self._adr_success_buffer) < self.adr_buffer_size:
            return

        rate = float(np.mean(self._adr_success_buffer))
        self._adr_success_buffer = []

        # Dynamic padding: at least 0.5 kg or 25 % of the total range.
        # Prevents delta from collapsing to zero when centre nears a limit.
        total_range = self.mass_max_limit - self.mass_min_limit
        padding = min(0.5, total_range / 4.0)

        # Shift centre in the direction of increasing difficulty
        self.adr_center = float(np.clip(
            self.adr_center + self.adr_center_lr * (rate - self.adr_success_threshold),
            self.mass_min_limit + padding,
            self.mass_max_limit - padding,
        ))

        if rate >= self.adr_success_threshold:
            # Agent is performing well → widen the randomization range
            new_delta = self.adr_delta + self.adr_expand_step
            max_reachable = min(
                self.adr_center - self.mass_min_limit,
                self.mass_max_limit - self.adr_center,
            )
            self.adr_delta = min(new_delta, max(max_reachable, 0.0))
            action = "EXPAND"
        else:
            # Agent is struggling → narrow the range, keep a minimum width
            self.adr_delta = max(0.10, self.adr_delta - self.adr_shrink_step)
            action = "SHRINK"

        # Store snapshot for the ADR evolution plot
        self.adr_history.append((self._episode_count, self.adr_center, self.adr_delta))
        print(
            f"[ADR] ep={self._episode_count:5d}  rate={rate:.2f}  {action}"
            f"  center={self.adr_center:.3f}"
            f"  range=[{self.adr_mass_low:.3f}, {self.adr_mass_high:.3f}]"
        )

    # Gymnasium overrides
    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        # Optionally, you can add here extra logic

        # Track success across the whole episode so ADR has a reliable signal
        # even when panda-gym omits is_success on the final terminal step
        if info.get("is_success", False):
            self._current_ep_success = True

        return obs, reward, terminated, truncated, info

    # Reset
    def reset(self, **kwargs):

        # Flush  ADR update from the episode that just finished
        if self.mode == "adr" and self._episode_count > 0:
            self._adr_update_range(self._current_ep_success)

        # Reset per-episode success flag and increment counter
        self._current_ep_success = False
        self._episode_count += 1

        new_mass = self._sample_mass()   # sample new mass

        if new_mass is not None:

            # Inject the new mass into the PyBullet simulation BEFORE
            # super().reset() so the very first observation already reflects
            # the correct physics (panda-gym does not call changeDynamics
            # internally during reset, so this mass persists through the call)
            sim = self.env.unwrapped.task.sim
            object_body_id = sim._bodies_idx["object"]

            sim.physics_client.changeDynamics(
                bodyUniqueId=object_body_id,
                linkIndex=-1,
                mass=float(new_mass),
            )

            self.last_mass = new_mass
            # Per-reset print disabled: thousands of resets per run causes
            # significant I/O overhead and log clutter.
            # ADR updates still print at every buffer flush (every
            # adr_buffer_size episodes).
        return super().reset(**kwargs)