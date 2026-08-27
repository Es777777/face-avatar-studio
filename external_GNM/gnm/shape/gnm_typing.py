# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typing compatibility helper for GNM and etils.enp.

This module provides `enpt`, a proxy wrapper around `etils.enp.typing`.

Why this module exists:
`etils.enp.typing` array annotations (such as `enpt.FloatArray['...']`) evaluate
to instances of `ArrayAliasMeta` at runtime. Static type checkers (e.g. Pyrefly)
flag metaclass instances in type annotations as non-type forms (`[not-a-type]`).

`_EnptWrapper` dynamically forwards attribute lookups to `etils.enp.typing` at
runtime while annotating `__getattr__` return type as `Any`. This allows static
type checkers to treat `enpt.<ArrayAlias>` as `Any` (avoiding `[not-a-type]`
build errors) while preserving `ArrayAliasMeta` instances at runtime for
`@enp.check_and_normalize_arrays`.
"""

from typing import Any
from etils import enp


class _EnptWrapper:
  """Wrapper around etils.enp.typing returning Any for typecheckers."""

  def __getattr__(self, name: str) -> Any:
    return getattr(enp.typing, name)


enpt = _EnptWrapper()
