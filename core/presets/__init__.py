from typing import Dict, List
from core.presets.base import BasePreset
from core.presets.customer_churn import CustomerChurnPreset
from core.presets.ecommerce import EcommercePreset
from core.presets.house_price import HousePricePreset

PRESET_REGISTRY: Dict[str, BasePreset] = {
    "customer_churn": CustomerChurnPreset(),
    "ecommerce": EcommercePreset(),
    "house_price": HousePricePreset(),
}


def get_preset(name: str) -> BasePreset:
    if name not in PRESET_REGISTRY:
        available = ", ".join(sorted(PRESET_REGISTRY.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available presets: {available}")
    return PRESET_REGISTRY[name]


def get_available_presets() -> List[str]:
    return list(PRESET_REGISTRY.keys())
