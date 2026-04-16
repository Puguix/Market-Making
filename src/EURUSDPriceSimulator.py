from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


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
    """

    def __init__(
        self,
        s0: float = 1.0850,
        dt_seconds: float = 0.01,
        bar_sigma_pips: float = 5.0,
        lambda_jump_per_day: float = 4.0,
        sigma_jump_pips: float = 7.5,
        seed: Optional[int] = None
    ):
        self.s0 = s0
        self.dt_seconds = dt_seconds
        self.bar_sigma_pips = bar_sigma_pips
        self.lambda_jump_per_day = lambda_jump_per_day
        self.sigma_jump_pips = sigma_jump_pips

        self._eps_B: float = 0.0  # current state AR(1) for B
        self._eps_C: float = 0.0  # current state AR(1) for C
        self._rho_eps: float = 0.9
        self._sigma_eps_pips: float = 0.3  # 

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

    def generate(self, n_steps: int) -> None:
        """
        Generate and append n_steps new prices to the FIFO buffers.

        This implementation is vectorised with NumPy to minimise Python
        overhead when simulating long paths (e.g., 24h at 50 ms).
        """
        if n_steps <= 0:
            return

        dt_s = self.dt_seconds
        dt_h = dt_s / 3600.0

        # Time grid for the new steps (in hours)
        idx = np.arange(1, n_steps + 1, dtype=float)
        t_seconds = self._t_seconds + idx * dt_s
        t_hours = t_seconds / 3600.0

        # Intraday seasonal activity φ(t) as in _session_activity, but vectorised
        h = np.mod(t_hours, 24.0)
        phi = np.full_like(h, 0.5)  # Overnight baseline
        mask_tokyo = (h >= 0.0) & (h < 8.0)
        mask_london_open = (h >= 8.0) & (h < 9.0)
        mask_london_mid = (h >= 9.0) & (h < 13.0)
        mask_overlap = (h >= 13.0) & (h < 16.0)
        mask_post_london = (h >= 16.0) & (h < 18.0)

        phi[mask_tokyo] = 0.6
        phi[mask_london_open] = 1.4
        phi[mask_london_mid] = 1.0
        phi[mask_overlap] = 1.5
        phi[mask_post_london] = 0.8

        sigma_pips = self.bar_sigma_pips * phi

        # Diffusion component in pips
        brownian_pips = sigma_pips * np.sqrt(dt_h) * self._rng.standard_normal(n_steps)

        # Macro jump component (Poisson arrivals with Gaussian jump sizes), in pips
        p_jump = self.lambda_jump_per_day * dt_s / (24.0 * 3600.0)
        jump_flags = self._rng.random(n_steps) < p_jump
        jump_sizes = self.sigma_jump_pips * self._rng.standard_normal(n_steps)
        jump_pips = np.where(jump_flags, jump_sizes, 0.0)

        total_pips = brownian_pips + jump_pips
        delta_price = total_pips / 10_000.0  # convert pips to price units

        # Build the base mid-price path over the new steps
        base_path = self._current_S + np.cumsum(delta_price)

        # Exchange-specific micro noise so B and C are not exactly equal
        innovations_B = self._sigma_eps_pips * self._rng.standard_normal(n_steps)
        innovations_C = self._sigma_eps_pips * self._rng.standard_normal(n_steps)

        # Construire le chemin AR(1) vectorisé
        noise_B_pips = np.zeros(n_steps)
        noise_C_pips = np.zeros(n_steps)
        eps_B, eps_C = self._eps_B, self._eps_C
        for i in range(n_steps):
            eps_B = self._rho_eps * eps_B + innovations_B[i]
            eps_C = self._rho_eps * eps_C + innovations_C[i]
            noise_B_pips[i] = eps_B
            noise_C_pips[i] = eps_C

        # Mettre à jour l'état pour le prochain appel
        self._eps_B = eps_B
        self._eps_C = eps_C

        s_B = base_path + noise_B_pips / 10_000.0
        s_C = base_path + noise_C_pips / 10_000.0

        # Append at the tail of FIFO buffers
        self._base_mid.extend(base_path.tolist())
        self._mid_B.extend(s_B.tolist())
        self._mid_C.extend(s_C.tolist())

        # Advance internal time and current price
        self._current_S = float(base_path[-1])
        self._t_seconds += n_steps * dt_s

    def next_prices(self) -> Tuple[float, float, float]:
        """
        Retrieve the next (base, B, C) mid-prices from the FIFO buffers.
        If the buffer is empty, one new step is generated on the fly.
        """
        if not self._base_mid:
            self._step()
        base = self._base_mid.popleft()
        b = self._mid_B.popleft()
        c = self._mid_C.popleft()
        return base, b, c

    def peek_next(self) -> Tuple[float, float, float]:
        """
        Inspect the next (base, B, C) mid-prices without consuming them.
        If the buffer is empty, one new step is generated first.
        """
        if not self._base_mid:
            self._step()
        base = self._base_mid[0]
        b = self._mid_B[0]
        c = self._mid_C[0]
        return base, b, c

    # ---------- core simulation ----------
    def _step(self) -> None:
        """Simulate one Euler step of the jump-diffusion mid-price."""

        dt_s = self.dt_seconds
        dt_h = dt_s / 3600.0
        t_hours = self._t_seconds / 3600.0

        # Intraday seasonal volatility σ_t = \bar{σ} · φ(t)
        phi_t = self._session_activity(t_hours)
        sigma_pips = self.bar_sigma_pips * phi_t

        # Diffusion component in pips
        brownian_pips = sigma_pips * np.sqrt(dt_h) * self._rng.standard_normal()

        # Macro jump component (Poisson arrivals with Gaussian jump sizes), in pips
        jump_pips = 0.0
        p_jump = self.lambda_jump_per_day * dt_s / (24.0 * 3600.0)
        if self._rng.random() < p_jump:
            jump_pips = self.sigma_jump_pips * self._rng.standard_normal()

        delta_pips = brownian_pips + jump_pips
        delta_price = delta_pips / 10_000.0  # convert pips to price units

        self._current_S += delta_price
        self._t_seconds += dt_s

        # Exchange-specific micro noise so B and C are not exactly equal, AR(1)
        self._eps_B = self._rho_eps * self._eps_B + self._sigma_eps_pips * self._rng.standard_normal()
        self._eps_C = self._rho_eps * self._eps_C + self._sigma_eps_pips * self._rng.standard_normal()
        noise_B_pips = self._eps_B
        noise_C_pips = self._eps_C

        noise_B = noise_B_pips / 10_000.0
        noise_C = noise_C_pips / 10_000.0

        s_B = self._current_S + noise_B
        s_C = self._current_S + noise_C

        # Append at the tail of FIFO buffers
        self._base_mid.append(self._current_S)
        self._mid_B.append(s_B)
        self._mid_C.append(s_C)

    @staticmethod
    def _session_activity(hour_of_day: float) -> float:
        """
        Intraday seasonal activity factor φ(t) from the table in eq. (2),
        using UTC hour of day.
        """
        h = hour_of_day % 24.0
        if 0.0 <= h < 8.0:   # Tokyo
            return 0.6
        if 8.0 <= h < 9.0:   # London open
            return 1.4
        if 9.0 <= h < 13.0:  # London mid
            return 1.0
        if 13.0 <= h < 16.0:  # London–NY overlap
            return 1.5
        if 16.0 <= h < 18.0:  # Post-London
            return 0.8
        return 0.5  # Overnight


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    # 24 hours with 50 ms time step
    dt = 0.05  # seconds
    total_seconds = 24 * 3600
    n_steps = int(total_seconds / dt)

    sim = EURUSDPriceSimulator(s0=1.0850, dt_seconds=dt, seed=42)

    t0 = time.perf_counter()
    sim.generate(n_steps)
    t1 = time.perf_counter()

    print(f"Simulated {n_steps} steps in {t1 - t0:.3f} seconds")

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

