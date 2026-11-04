"""CellWorld model code, vendored unchanged except for relative imports.

Source: github.com/UoM-HealthAI/CellWorld at commit c40203f398342dbe148bcca9891a8d58bf40070a, files model/{cellworld,encoder,
tokenizer,pos_encoding}.py. MIT License, Copyright (c) 2026 Haiping Liu (see LICENSE in this folder). Vendored because the upstream
package installs top-level modules named `data` and `model`."""
from .cellworld import CellWorld
from .pos_encoding import normalize_positions
