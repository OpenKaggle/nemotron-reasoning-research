# Deterministic Solver Baseline

This report evaluates local deterministic solvers used to generate SFT traces.

## Accuracy

| Family | Correct | Total | Accuracy |
| --- | ---: | ---: | ---: |
| bit_manipulation | 1440 | 1602 | 89.89% |
| gravity_physics | 1597 | 1597 | 100.00% |
| numeral_system | 1576 | 1576 | 100.00% |
| symbolic_equation | 581 | 1555 | 37.36% |
| text_cipher | 1576 | 1576 | 100.00% |
| unit_conversion | 1594 | 1594 | 100.00% |
| **overall** | **8364** | **9500** | **88.04%** |

## Notes

- Cipher vocabulary size from prompt examples: 77.
- Numeral, gravity, and unit conversion are solved directly from parsed structure.
- Text cipher uses prompt-derived substitution mappings plus global Wonderland vocabulary fill.
- Bit manipulation now prioritizes global ROT/SHL/SHR expressions with up to three transforms, then falls back to the per-output-bit gate matcher.
- Symbolic equation remains the main unsolved family; numeric equations are partially solved, while cipher/symbol-digit equations need a stronger mapping search.
