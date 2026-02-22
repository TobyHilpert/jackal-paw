from __future__ import annotations

import numpy as np


class LinearMixer:
    def __init__(self, beta: float = 0.4):
        self.beta = beta

    def mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        return (1.0 - self.beta) * rho_in + self.beta * rho_out


class DIISMixer:
    """Pulay/DIIS mixer with lightweight Kerker-like preconditioning."""

    def __init__(self, beta: float = 0.4, ndim: int = 8, kerker_q0: float = 1.0):
        self.beta = beta
        self.ndim = ndim
        self.kerker_q0 = kerker_q0
        self.history: list[tuple[np.ndarray, np.ndarray]] = []
        self._fallback = LinearMixer(beta=beta)

    def _kerker_precondition(self, resid: np.ndarray) -> np.ndarray:
        if resid.ndim != 3:
            return resid
        qx = np.fft.fftfreq(resid.shape[0])[:, None, None]
        qy = np.fft.fftfreq(resid.shape[1])[None, :, None]
        qz = np.fft.fftfreq(resid.shape[2])[None, None, :]
        q2 = qx * qx + qy * qy + qz * qz
        filt = q2 / (q2 + self.kerker_q0 * self.kerker_q0)
        r_g = np.fft.fftn(resid)
        return np.fft.ifftn(r_g * filt).real

    def mix(self, rho_in: np.ndarray, rho_out: np.ndarray) -> np.ndarray:
        resid = self._kerker_precondition(rho_out - rho_in)
        self.history.append((rho_in.copy(), resid.copy()))
        self.history = self.history[-self.ndim :]

        if len(self.history) < 2:
            return self._fallback.mix(rho_in, rho_in + resid)

        r = np.array([h[1].ravel() for h in self.history])
        b = np.zeros((len(self.history) + 1, len(self.history) + 1))
        b[:-1, :-1] = r @ r.T
        b[:-1, -1] = -1.0
        b[-1, :-1] = -1.0
        rhs = np.zeros(len(self.history) + 1)
        rhs[-1] = -1.0

        try:
            coeff = np.linalg.solve(b + 1e-12 * np.eye(b.shape[0]), rhs)[:-1]
            mixed = np.sum([c * (x + e) for c, (x, e) in zip(coeff, self.history)], axis=0)
            return (1.0 - self.beta) * rho_in + self.beta * mixed
        except np.linalg.LinAlgError:
            return self._fallback.mix(rho_in, rho_in + resid)
