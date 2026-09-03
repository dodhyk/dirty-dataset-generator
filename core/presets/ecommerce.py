from typing import List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from core.models import ColumnRole, ColumnSpec, DataType
from core.presets.base import BasePreset
from core.distributions import (
    generate_unique_ids,
    sample_truncated_normal_int,
    sample_truncated_normal_float,
    sample_weighted_categorical,
)


class EcommercePreset(BasePreset):
    @property
    def name(self) -> str:
        return "ecommerce"

    def get_schema(self) -> List[ColumnSpec]:
        return [
            ColumnSpec("customer_id", DataType.STRING, ColumnRole.ID, is_protected=True),
            ColumnSpec("order_id", DataType.STRING, ColumnRole.TRANSACTION_ID, is_protected=False),
            ColumnSpec("product_id", DataType.STRING, ColumnRole.PRODUCT_ID, is_protected=False),
            ColumnSpec("product_name", DataType.STRING, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("age", DataType.INTEGER, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("gender", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("location", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("device_type", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("membership_type", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("total_orders", DataType.INTEGER, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("total_spent", DataType.DECIMAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("avg_order_value", DataType.DECIMAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("items_per_order", DataType.DECIMAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("days_since_last_order", DataType.INTEGER, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("discount_usage", DataType.INTEGER, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("preferred_category", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("payment_method", DataType.CATEGORICAL, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("timestamp", DataType.TIMESTAMP, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("shipping_address", DataType.STRING, ColumnRole.FEATURE, is_protected=False),
            ColumnSpec("purchased_again", DataType.INTEGER, ColumnRole.TARGET, is_protected=True),
        ]

    def generate_clean(self, rows: int, rng: np.random.Generator) -> pd.DataFrame:
        customer_ids = generate_unique_ids("CUST", rows, digits=6)
        order_ids = generate_unique_ids("ORD", rows, digits=7)

        category_products = {
            "Electronics": [
                ("PROD-101", "Wireless Noise-Canceling Headphones"),
                ("PROD-102", "Smartwatch Fitness Tracker"),
                ("PROD-103", "Portable Bluetooth Speaker"),
                ("PROD-104", "USB-C Fast Charging Hub"),
            ],
            "Apparel": [
                ("PROD-201", "Cotton Crewneck T-Shirt"),
                ("PROD-202", "Slim Fit Denim Jeans"),
                ("PROD-203", "Lightweight Running Jacket"),
                ("PROD-204", "Breathable Athletic Socks"),
            ],
            "Home & Kitchen": [
                ("PROD-301", "Stainless Steel French Press"),
                ("PROD-302", "Non-Stick Ceramic Frying Pan"),
                ("PROD-303", "Ergonomic Memory Foam Pillow"),
                ("PROD-304", "Electric Gooseneck Kettle"),
            ],
            "Beauty & Personal Care": [
                ("PROD-401", "Hydrating Hyaluronic Acid Serum"),
                ("PROD-402", "SPF 50 Mineral Sunscreen"),
                ("PROD-403", "Nourishing Argan Hair Oil"),
                ("PROD-404", "Gentle Foaming Facial Cleanser"),
            ],
            "Books": [
                ("PROD-501", "Designing Data-Intensive Applications"),
                ("PROD-502", "Atomic Habits by James Clear"),
                ("PROD-503", "Python Data Science Handbook"),
                ("PROD-504", "Clean Architecture"),
            ],
            "Sports": [
                ("PROD-601", "Non-Slip Exercise Yoga Mat"),
                ("PROD-602", "Adjustable Dumbbell Set"),
                ("PROD-603", "Insulated Sports Water Bottle"),
                ("PROD-604", "Resistance Bands Set"),
            ],
        }

        categories = list(category_products.keys())
        preferred_categories = sample_weighted_categorical(
            rng, categories, [0.25, 0.22, 0.18, 0.15, 0.10, 0.10], rows
        )

        product_ids = []
        product_names = []
        for cat in preferred_categories:
            prods = category_products[cat]
            idx = rng.integers(0, len(prods))
            p_id, p_name = prods[idx]
            product_ids.append(p_id)
            product_names.append(p_name)

        ages = sample_truncated_normal_int(rng, mean=34, std=11, low=18, high=70, size=rows)
        genders = sample_weighted_categorical(rng, ["Female", "Male", "Other"], [0.52, 0.45, 0.03], rows)
        locations = sample_weighted_categorical(
            rng,
            ["Jakarta", "Surabaya", "Bandung", "Medan", "Semarang", "Makassar", "Yogyakarta", "Denpasar"],
            [0.35, 0.18, 0.15, 0.10, 0.08, 0.06, 0.05, 0.03],
            rows,
        )
        device_types = sample_weighted_categorical(
            rng, ["Mobile App", "Desktop Web", "Mobile Web"], [0.65, 0.25, 0.10], rows
        )
        membership_types = sample_weighted_categorical(
            rng, ["Regular", "Silver", "Gold", "Platinum"], [0.50, 0.25, 0.15, 0.10], rows
        )

        total_orders = sample_truncated_normal_int(rng, mean=7, std=6, low=1, high=50, size=rows)
        items_per_order = sample_truncated_normal_float(rng, mean=2.6, std=1.2, low=1.0, high=10.0, decimals=1, size=rows)

        # avg_order_value conditioned on membership
        membership_mult = {
            "Regular": 1.0,
            "Silver": 1.25,
            "Gold": 1.6,
            "Platinum": 2.2,
        }
        avg_order_value = np.zeros(rows, dtype=float)
        for i in range(rows):
            mult = membership_mult[membership_types[i]]
            base = sample_truncated_normal_float(rng, mean=42.0 * mult, std=15.0 * mult, low=10.0, high=350.0, decimals=2, size=1)[0]
            avg_order_value[i] = base

        # total_spent = total_orders * avg_order_value
        total_spent = np.round(total_orders * avg_order_value, 2)

        days_since_last_order = sample_truncated_normal_int(rng, mean=28, std=24, low=0, high=180, size=rows)

        # discount_usage: orders with discounts applied (0 <= discount_usage <= total_orders)
        discount_usage = np.zeros(rows, dtype=int)
        for i in range(rows):
            discount_usage[i] = int(rng.integers(0, total_orders[i] + 1))

        payment_methods = sample_weighted_categorical(
            rng,
            ["E-Wallet", "Credit Card", "Bank Transfer", "Cash on Delivery", "Buy Now Pay Later"],
            [0.35, 0.25, 0.20, 0.12, 0.08],
            rows,
        )

        # Realistic timestamps in 2026
        base_time = datetime(2026, 1, 1, 0, 0, 0)
        random_seconds = rng.integers(0, 240 * 24 * 3600, size=rows)
        timestamps = [(base_time + timedelta(seconds=int(s))).strftime("%Y-%m-%dT%H:%M:%S") for s in random_seconds]

        # Shipping addresses
        streets = [
            "Jl. Sudirman", "Jl. Thamrin", "Jl. Gatot Subroto", "Jl. Asia Afrika",
            "Jl. Malioboro", "Jl. Diponegoro", "Jl. Ahmad Yani", "Jl. Pemuda",
        ]
        shipping_addresses = []
        for i in range(rows):
            street = streets[rng.integers(0, len(streets))]
            num = rng.integers(1, 200)
            loc = locations[i]
            shipping_addresses.append(f"{street} No. {num}, {loc}")

        # Behavioral Target Relationship: purchased_again (Classification)
        # Higher total_orders, lower days_since_last_order, higher membership -> high probability
        mem_score = np.array([{"Regular": 0, "Silver": 1, "Gold": 2, "Platinum": 3}[m] for m in membership_types])
        log_odds = -0.5
        log_odds += 0.12 * total_orders
        log_odds -= 0.035 * days_since_last_order
        log_odds += 0.45 * mem_score
        log_odds += 0.001 * total_spent
        log_odds += rng.normal(loc=0.0, scale=0.3, size=rows)

        prob_purchase = 1.0 / (1.0 + np.exp(-log_odds))
        purchased_again = (rng.random(size=rows) < prob_purchase).astype(int)

        return pd.DataFrame(
            {
                "customer_id": customer_ids,
                "order_id": order_ids,
                "product_id": product_ids,
                "product_name": product_names,
                "age": ages,
                "gender": genders,
                "location": locations,
                "device_type": device_types,
                "membership_type": membership_types,
                "total_orders": total_orders,
                "total_spent": total_spent,
                "avg_order_value": avg_order_value,
                "items_per_order": items_per_order,
                "days_since_last_order": days_since_last_order,
                "discount_usage": discount_usage,
                "preferred_category": preferred_categories,
                "payment_method": payment_methods,
                "timestamp": timestamps,
                "shipping_address": shipping_addresses,
                "purchased_again": purchased_again,
            }
        )
