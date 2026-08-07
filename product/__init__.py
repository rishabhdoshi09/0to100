"""Retail product projections. Backend trading state remains authoritative."""
from product.gather import gather_product_inputs
from product.projection import ProductInputs, ProductState, SetupStep, TERMINOLOGY, build_product_state

__all__ = [
    "ProductInputs", "ProductState", "SetupStep", "TERMINOLOGY",
    "build_product_state", "gather_product_inputs",
]
