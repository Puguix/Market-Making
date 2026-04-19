from collections import deque
from typing import Deque, Optional, Tuple
import numpy as np
from scipy.signal import lfilter

from config import (
    PRICE_SIM_DEFAULT_S0, PRICE_SIM_DEFAULT_DT_SECONDS,
    PRICE_SIM_DEFAULT_BAR_SIGMA_PIPS, PRICE_SIM_DEFAULT_LAMBDA_JUMP_PER_DAY,
    PRICE_SIM_DEFAULT_SIGMA_JUMP_PIPS, PRICE_SIM_DEFAULT_RHO_EPS,
    PRICE_SIM_DEFAULT_SIGMA_EPS_PIPS, HOURS_PER_DAY, SECONDS_PER_HOUR,
    SECONDS_PER_DAY, PRICE_SIM_DEFAULT_DAY_STEPS, TOKYO_ACTIVITY,
    LONDON_OPEN_ACTIVITY, LONDON_MID_ACTIVITY, OVERLAP_ACTIVITY,
    POST_LONDON_ACTIVITY, OVERNIGHT_ACTIVITY, PIPS_TO_PRICE_SCALE
)

class EURUSDPriceSimulator:
    """
    Discrete-time EUR/USD mid-price simulator based on the jump-diffusion model of 2.1.

    - Underlying mid-price S_t follows:
        dS_t = σ_t dW_t + J_t dN_t
      with intraday seasonal volatility σ_t and occasional macro jumps.
    - Prices for exchanges B and C are obtained by adding small independent noise
      to the same underlying mid-price.
    - Generated prices are buffered in FIFO deques so reading from the head and
      appending at the tail are both O(1).

    Parameters:
    - s0: initial mid-price
    - dt_seconds: time step in seconds
    - bar_sigma_pips: annualised volatility in pips
    - lambda_jump_per_day: average number of jumps per day
    - sigma_jump_pips: standard deviation of jump sizes in pips
    - seed: random seed for reproducibility

    Attributes:
    - _eps_B: current state AR(1) for B
    - _eps_C: current state AR(1) for C
    - _rho_eps: correlation coefficient for AR(1)
    - _sigma_eps_pips: standard deviation of AR(1) noise in pips
    - _rng: random number generator
    - _t_seconds: simulation time in seconds after the last consumed price step
    - _current_S: current mid-price
    - _base_mid: FIFO buffer for base mid-price
    - _mid_B: FIFO buffer for B mid-price
    - _mid_C: FIFO buffer for C mid-price
    """

    def __init__(
        self,
        s0: float = PRICE_SIM_DEFAULT_S0,
        dt_seconds: float = PRICE_SIM_DEFAULT_DT_SECONDS,
        bar_sigma_pips: float = PRICE_SIM_DEFAULT_BAR_SIGMA_PIPS,
        lambda_jump_per_day: float = PRICE_SIM_DEFAULT_LAMBDA_JUMP_PER_DAY,
        sigma_jump_pips: float = PRICE_SIM_DEFAULT_SIGMA_JUMP_PIPS,
        seed: Optional[int] = None
    ):
        self.s0 = s0
        self.dt_seconds = dt_seconds
        self.bar_sigma_pips = bar_sigma_pips
        self.lambda_jump_per_day = lambda_jump_per_day
        self.sigma_jump_pips = sigma_jump_pips

        self._eps_B: float = 0.0  # current state AR(1) for B
        self._eps_C: float = 0.0  # current state AR(1) for C
        self._rho_eps: float = PRICE_SIM_DEFAULT_RHO_EPS
        self._sigma_eps_pips: float = PRICE_SIM_DEFAULT_SIGMA_EPS_PIPS

        self._rng = np.random.default_rng(seed)

        self._t_seconds: float = 0.0
        self._current_S: float = s0

        # FIFO buffers for future prices (base, B, C)
        self._base_mid: Deque[float] = deque()
        self._mid_B: Deque[float] = deque()
        self._mid_C: Deque[float] = deque()

    # ---------- public API ----------
    def buffer_size(self) -> int:
        """Current number of buffered price points."""
        return len(self._base_mid)

    def generate_prices(self, n_steps: int = PRICE_SIM_DEFAULT_DAY_STEPS) -> None:
        """
        Generate and append new prices for a simulation day to the FIFO buffers.

        This implementation is vectorised with NumPy to minimise Python
        overhead when simulating long paths (e.g., 24h at 10ms time step).

        Generate 8_640_000 steps by default, which is 24H at 10ms time step.
        """

        dt_s = self.dt_seconds
        dt_h = dt_s / SECONDS_PER_HOUR

        # Time grid for the new steps (in hours)
        idx = np.arange(1, n_steps + 1, dtype=float)
        t_seconds = self._t_seconds + idx * dt_s
        t_hours = t_seconds / SECONDS_PER_HOUR

        # Intraday seasonal activity φ(t) as in _session_activity, but vectorised
        h = np.mod(t_hours, HOURS_PER_DAY)
        phi = np.full_like(h, OVERNIGHT_ACTIVITY)  # Overnight baseline
        mask_tokyo = (h >= 0.0) & (h < 8.0)
        mask_london_open = (h >= 8.0) & (h < 9.0)
        mask_london_mid = (h >= 9.0) & (h < 13.0)
        mask_overlap = (h >= 13.0) & (h < 16.0)
        mask_post_london = (h >= 16.0) & (h < 18.0)

        phi[mask_tokyo] = TOKYO_ACTIVITY
        phi[mask_london_open] = LONDON_OPEN_ACTIVITY
        phi[mask_london_mid] = LONDON_MID_ACTIVITY
        phi[mask_overlap] = OVERLAP_ACTIVITY
        phi[mask_post_london] = POST_LONDON_ACTIVITY

        sigma_pips = self.bar_sigma_pips * phi

        # Diffusion component in pips
        brownian_pips = sigma_pips * np.sqrt(dt_h) * self._rng.standard_normal(n_steps)

        # Macro jump component (Poisson arrivals with Gaussian jump sizes), in pips
        p_jump = self.lambda_jump_per_day * dt_s / (HOURS_PER_DAY * SECONDS_PER_HOUR)
        jump_flags = self._rng.random(n_steps) < p_jump
        jump_sizes = self.sigma_jump_pips * self._rng.standard_normal(n_steps)
        jump_pips = np.where(jump_flags, jump_sizes, 0.0)

        total_pips = brownian_pips + jump_pips
        delta_price = total_pips / PIPS_TO_PRICE_SCALE  # convert pips to price units

        # Build the base mid-price path over the new steps
        base_path = self._current_S + np.cumsum(delta_price)

        # Exchange-specific micro noise so B and C are not exactly equal
        innovations_B = self._sigma_eps_pips * self._rng.standard_normal(n_steps)
        innovations_C = self._sigma_eps_pips * self._rng.standard_normal(n_steps)

        # Vectorized AR(1): eps_t = rho * eps_{t-1} + innovation_t
        rho = self._rho_eps
        noise_B_pips = lfilter([1], [1, -rho], innovations_B, zi=[self._eps_B])[0]
        noise_C_pips = lfilter([1], [1, -rho], innovations_C, zi=[self._eps_C])[0]

        # Update AR(1) state for the next call
        self._eps_B = noise_B_pips[-1]
        self._eps_C = noise_C_pips[-1]

        s_B = base_path + noise_B_pips / PIPS_TO_PRICE_SCALE
        s_C = base_path + noise_C_pips / PIPS_TO_PRICE_SCALE

        # Append at the tail of FIFO buffers
        self._base_mid.extend(base_path.tolist())
        self._mid_B.extend(s_B.tolist())
        self._mid_C.extend(s_C.tolist())

        # Advance path state for the next generate_prices batch; simulation clock _t_seconds
        # moves only when prices are consumed in next_prices() so per-step metrics timestamps differ.
        self._current_S = float(base_path[-1])

    def next_prices(self) -> Tuple[float, float, float]:
        """
        Retrieve the next (base, B, C) mid-prices from the FIFO buffers.
        Advances simulation time by one dt. Buffer must be non-empty.
        """
        base = self._base_mid.popleft()
        mid_b = self._mid_B.popleft()
        mid_c = self._mid_C.popleft()
        self._t_seconds += self.dt_seconds
        return base, mid_b, mid_c

    def peek_next(self) -> Tuple[float, float, float]:
        """
        Inspect the next (base, B, C) mid-prices without consuming them.
        Buffer must be non-empty.
        """
        return self._base_mid[0], self._mid_B[0], self._mid_C[0]

    @staticmethod
    def _session_activity(hour_of_day: float) -> float:
        """
        Intraday seasonal activity factor φ(t) from the table in eq. (2),
        using UTC hour of day.
        """
        h = hour_of_day % HOURS_PER_DAY
        if 0.0 <= h < 8.0:
            return TOKYO_ACTIVITY  # Tokyo
        if 8.0 <= h < 9.0:
            return LONDON_OPEN_ACTIVITY  # London open
        if 9.0 <= h < 13.0:
            return LONDON_MID_ACTIVITY  # London mid
        if 13.0 <= h < 16.0:
            return OVERLAP_ACTIVITY  # London–NY overlap
        if 16.0 <= h < 18.0:
            return POST_LONDON_ACTIVITY  # Post-London
        return OVERNIGHT_ACTIVITY  # Overnight

    def generate_prices_for_simulation_day(self) -> None:
        """
        Generate and append new prices for a simulation day to the FIFO buffers.
        """
        self.generate_prices(int(SECONDS_PER_DAY / self.dt_seconds))


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    # 50 ms time step
    dt = 0.05  # seconds

    sim = EURUSDPriceSimulator(s0=1.0850, dt_seconds=dt, seed=42)

    t0 = time.perf_counter()
    sim.generate_prices_for_simulation_day()
    t1 = time.perf_counter()

    print(f"Simulated 24 hours in {t1 - t0:.3f} seconds")

    # Extract full mid-price paths from the FIFO without extra stepping
    base_prices = list(sim._base_mid)
    b_prices = list(sim._mid_B)
    c_prices = list(sim._mid_C)
    time_axis_hours = np.linspace(0.0, 24.0, num=len(base_prices), endpoint=False)

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis_hours, base_prices, label="Base mid-price", linewidth=1.2)
    plt.plot(time_axis_hours, b_prices, label="Exchange B mid-price", alpha=0.8, linewidth=0.9)
    plt.plot(time_axis_hours, c_prices, label="Exchange C mid-price", alpha=0.8, linewidth=0.9)
    plt.xlabel("Time (hours)")
    plt.ylabel("EUR/USD mid-price")
    plt.title("EUR/USD mid-price evolution over 24 hours (base, B, C)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

