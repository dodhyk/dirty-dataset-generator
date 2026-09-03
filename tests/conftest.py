import pytest
from core.generator import generate
from core.models import GenerationResult

BASELINE_SEED = 12345
BASELINE_ROWS = 1000
BASELINE_DIRTINESS = 20.0


@pytest.fixture(scope="session")
def baseline_seed() -> int:
    return BASELINE_SEED


@pytest.fixture(scope="session")
def churn_gen_1k() -> GenerationResult:
    return generate(
        preset="customer_churn",
        rows=BASELINE_ROWS,
        dirtiness=BASELINE_DIRTINESS,
        seed=BASELINE_SEED,
    )


@pytest.fixture(scope="session")
def ecom_gen_1k() -> GenerationResult:
    return generate(
        preset="ecommerce",
        rows=BASELINE_ROWS,
        dirtiness=BASELINE_DIRTINESS,
        seed=BASELINE_SEED,
    )


@pytest.fixture(scope="session")
def house_gen_1k() -> GenerationResult:
    return generate(
        preset="house_price",
        rows=BASELINE_ROWS,
        dirtiness=BASELINE_DIRTINESS,
        seed=BASELINE_SEED,
    )
