"""標高別パワー減衰モデル(プラグイン方式・仕様書 9 節)。

空気密度側の効果(9.3 節)は physics.air_density で別に扱う。
ここでは有酸素パワーの低下のみを担当する。
"""

from __future__ import annotations

from typing import Callable, Dict

from .constants import DEFAULT_H_THRESHOLD, DEFAULT_K_DECAY


class AltitudeModel:
    """f(h): 標高 h [m] における持続可能パワーの倍率を返すインターフェース。"""

    name = "base"

    def factor(self, h: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class ThresholdLinear(AltitudeModel):
    """閾値以下は 1.0、超過分に線形減衰。"""

    name = "threshold_linear"

    def __init__(self, h_threshold: float = DEFAULT_H_THRESHOLD, k: float = DEFAULT_K_DECAY):
        self.h_threshold = h_threshold
        self.k = k

    def factor(self, h: float) -> float:
        if h <= self.h_threshold:
            return 1.0
        return max(0.0, 1.0 - self.k * (h - self.h_threshold))

    def describe(self) -> str:
        return f"{self.name}(h_threshold={self.h_threshold:.0f} m, k={self.k:.5f}/m)"


class BassettAltitude(AltitudeModel):
    """Bassett et al. (1999) の高度別 VO2max 低下を多項式近似したモデル。

    海抜 ~1500 m 未満は無視できる低下、それ以上で加速度的に低下する。
    """

    name = "bassett_poly"

    def __init__(self, h_threshold: float = 1500.0):
        self.h_threshold = h_threshold

    def factor(self, h: float) -> float:
        if h <= self.h_threshold:
            return 1.0
        dh_km = (h - self.h_threshold) / 1000.0
        # 約: 1 km で 7%、2 km で 16% 低下
        drop = 0.0680 * dh_km + 0.0090 * dh_km ** 2
        return max(0.0, 1.0 - drop)

    def describe(self) -> str:
        return f"{self.name}(h_threshold={self.h_threshold:.0f} m)"


_REGISTRY: Dict[str, Callable[..., AltitudeModel]] = {
    "threshold_linear": ThresholdLinear,
    "bassett_poly": BassettAltitude,
}


def available_models() -> list[str]:
    return list(_REGISTRY)


def make_model(name: str, **kwargs) -> AltitudeModel:
    if name not in _REGISTRY:
        raise KeyError(f"未知の標高補正モデル: {name}. 利用可能: {available_models()}")
    return _REGISTRY[name](**kwargs)
