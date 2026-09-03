from typing import Optional
import secrets
import numpy as np

from core.models import GenerationConfig, GenerationResult
from core.presets import get_preset
from core.corruption import apply_corruption


def generate(
    preset: str,
    rows: int,
    dirtiness: float,
    seed: Optional[int] = None,
) -> GenerationResult:
    """Generate clean and dirty datasets along with a ground truth corruption report.
    
    This function is completely independent of any web framework (FastAPI) or UI (Streamlit).
    """
    config = GenerationConfig(preset=preset, rows=rows, dirtiness=dirtiness, seed=seed)

    # Determine seed: use provided seed or securely generate a random 32-bit integer seed
    effective_seed = seed if seed is not None else (secrets.randbits(31))

    # Initialize seeded random generator
    rng = np.random.default_rng(effective_seed)

    # 1. Preset resolution
    preset_obj = get_preset(config.preset)

    # 2. Clean generation
    clean_df = preset_obj.generate_clean(rows=config.rows, rng=rng)

    # 3. Dirty generation & Corruption report
    dirty_df, report = apply_corruption(
        clean_df=clean_df,
        preset=preset_obj,
        dirtiness=config.dirtiness,
        seed=effective_seed,
        rng=rng,
    )

    return GenerationResult(
        preset=config.preset,
        clean_df=clean_df,
        dirty_df=dirty_df,
        report=report,
    )
