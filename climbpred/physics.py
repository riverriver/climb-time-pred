"""物理モデル:空気密度と基礎運動方程式。"""

from __future__ import annotations

import numpy as np

from .constants import (
    G,
    LAPSE,
    M_AIR,
    P0,
    R_SPECIFIC,
    R_UNIV,
    RHO0,
    T0,
)


def air_pressure(h):
    """標高 h [m] における気圧 [Pa](国際標準大気・対流圏)。"""
    h = np.asarray(h, dtype=float)
    return P0 * (1.0 - LAPSE * h / T0) ** (G * M_AIR / (R_UNIV * LAPSE))


def air_density(h, temp_c=None):
    """標高 h [m] における空気密度 [kg/m^3]。

    temp_c を与えた場合はその実測気温で理想気体則により補正する。
    与えない場合は国際標準大気(海面 15 degC)を用いる。
    """
    h = np.asarray(h, dtype=float)
    if temp_c is None:
        return RHO0 * (1.0 - LAPSE * h / T0) ** (G * M_AIR / (R_UNIV * LAPSE) - 1.0)
    temp_k = np.asarray(temp_c, dtype=float) + 273.15
    return air_pressure(h) / (R_SPECIFIC * temp_k)


def grade_to_angle(grade):
    """勾配(tan) を角度 [rad] に変換。"""
    return np.arctan(np.asarray(grade, dtype=float))


def power_at_speed(v, grade, mass, crr, cda, rho, eta):
    """速度 v [m/s] で走行するのに必要なライダー出力 [W]。

    P_rider = (M g sinθ v + Crr M g cosθ v + 0.5 CdA ρ v^3) / η
    """
    theta = grade_to_angle(grade)
    p_wheel = (
        mass * G * np.sin(theta) * v
        + crr * mass * G * np.cos(theta) * v
        + 0.5 * cda * rho * v ** 3
    )
    return p_wheel / eta


def speed_at_power(power, grade, mass, crr, cda, rho, eta, v_max=25.0):
    """ライダー出力 power [W] で走行したときの平衡速度 [m/s]。

    0.5 CdA ρ v^3 + (M g sinθ + Crr M g cosθ) v - η P = 0
    の正の実根を返す。
    """
    theta = grade_to_angle(grade)
    a3 = 0.5 * cda * rho
    a1 = mass * G * (np.sin(theta) + crr * np.cos(theta))
    a0 = -eta * power

    roots = np.roots([a3, 0.0, a1, a0])
    real = roots[np.abs(roots.imag) < 1e-6].real
    positive = real[real > 0]
    if positive.size == 0:
        # 数値的に根が取れない場合は Newton 法にフォールバック
        return _newton_speed(power, a3, a1, a0, v_max)
    return float(min(positive.min(), v_max))


def _newton_speed(power, a3, a1, a0, v_max):
    v = 5.0
    for _ in range(100):
        f = a3 * v ** 3 + a1 * v + a0
        df = 3 * a3 * v ** 2 + a1
        if abs(df) < 1e-12:
            break
        step = f / df
        v -= step
        if v <= 0:
            v = 0.1
        if abs(step) < 1e-6:
            break
    return float(min(max(v, 0.0), v_max))
