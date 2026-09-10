"""Project 02: Structured Output Engine.

Schema enforcement for model output: a hand-written JSON Schema subset
validator, a schema builder driven by Python type hints, a repair pipeline for
broken JSON, and a bounded retry loop that feeds validation errors back to the
model before falling back to a typed default.
"""
from .engine import Attempt, StructuredOutputEngine, StructuredResult
from .extraction import Extraction, extract_json
from .schema_builder import schema_from_dataclass, schema_from_type
from .validator import SchemaError, describe_errors, is_valid, validate

__all__ = [
    "Attempt", "StructuredOutputEngine", "StructuredResult",
    "Extraction", "extract_json",
    "schema_from_dataclass", "schema_from_type",
    "SchemaError", "describe_errors", "is_valid", "validate",
]
