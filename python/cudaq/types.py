# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #
"""
Type stubs for quantum register types used only within CUDA-Q kernels.
These types are never executed at Python runtime; kernel code is parsed
into MLIR and JIT-executed. Stub implementations raise RuntimeError if
accidentally used outside a kernel.
"""

from __future__ import annotations

from typing import Iterator, overload

_KERNEL_ONLY = "Can only be used within a CUDA-Q kernel."


class qubit:
    """
    The qubit is the primary unit of information in a quantum computer.
    Qubits can be created individually or as part of larger registers.
    """

    def __init__(self) -> None:
        raise RuntimeError(_KERNEL_ONLY)

    def __invert__(self) -> qubit:
        """Negate the control qubit."""
        raise RuntimeError(_KERNEL_ONLY)

    def is_negated(self) -> bool:
        """Returns true if this is a negated control qubit."""
        raise RuntimeError(_KERNEL_ONLY)

    def reset_negation(self) -> None:
        """Removes the negated state of a control qubit."""
        raise RuntimeError(_KERNEL_ONLY)

    def id(self) -> int:
        """Return a unique integer identifier for this qubit."""
        raise RuntimeError(_KERNEL_ONLY)


class qview:
    """A non-owning view on a register of qubits."""

    def size(self) -> int:
        """Return the number of qubits in this view."""
        raise RuntimeError(_KERNEL_ONLY)

    @overload
    def front(self) -> qubit:
        ...

    @overload
    def front(self, count: int) -> qview:
        ...

    def front(self, count: int | None = None) -> qubit | qview:
        """Return first qubit(s) in this view."""
        raise RuntimeError(_KERNEL_ONLY)

    @overload
    def back(self) -> qubit:
        ...

    @overload
    def back(self, count: int) -> qview:
        ...

    def back(self, count: int | None = None) -> qubit | qview:
        """Return the last qubit(s) in this view."""
        raise RuntimeError(_KERNEL_ONLY)

    def __iter__(self) -> Iterator[qubit]:
        raise RuntimeError(_KERNEL_ONLY)

    def slice(self, start: int, count: int) -> qview:
        """Return the [start, start+count] qudits as a non-owning `qview`."""
        raise RuntimeError(_KERNEL_ONLY)

    def __getitem__(self, idx: int) -> qubit:
        """Return the qubit at the given index."""
        raise RuntimeError(_KERNEL_ONLY)


class qvector:
    """
    An owning, dynamically sized container for qubits. The semantics of the
    `qvector` follows that of a `std::vector` or list for qubits.
    """

    def __init__(self, size: int) -> None:
        raise RuntimeError(_KERNEL_ONLY)

    def size(self) -> int:
        """Return the number of qubits in this `qvector`."""
        raise RuntimeError(_KERNEL_ONLY)

    @overload
    def front(self) -> qubit:
        ...

    @overload
    def front(self, count: int) -> qview:
        ...

    def front(self, count: int | None = None) -> qubit | qview:
        """Return first qubit(s) in this `qvector`."""
        raise RuntimeError(_KERNEL_ONLY)

    @overload
    def back(self) -> qubit:
        ...

    @overload
    def back(self, count: int) -> qview:
        ...

    def back(self, count: int | None = None) -> qubit | qview:
        """Return the last qubit(s) in this `qvector`."""
        raise RuntimeError(_KERNEL_ONLY)

    def __iter__(self) -> Iterator[qubit]:
        raise RuntimeError(_KERNEL_ONLY)

    def slice(self, start: int, count: int) -> qview:
        """Return the [start, start+count] qudits as a non-owning `qview`."""
        raise RuntimeError(_KERNEL_ONLY)

    def __getitem__(self, idx: int) -> qubit:
        """Return the qubit at the given index."""
        raise RuntimeError(_KERNEL_ONLY)
