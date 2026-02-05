/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

/// This header declares the NVQIR quantum instruction runtime API functions
/// used by the CUDA-Q compilation platform for local simulation.

// Forward declarations
class Array;
struct Qubit;
using Result = bool;

namespace cudaq {
class SimulationState;
class state;
} // namespace cudaq

extern "C" {

//===----------------------------------------------------------------------===//
// Initialization and Finalization
//===----------------------------------------------------------------------===//

/// @brief Initialize the QIR runtime.
void __quantum__rt__initialize(int argc, int8_t **argv);

/// @brief Finalize the NVQIR library.
void __quantum__rt__finalize();

//===----------------------------------------------------------------------===//
// Qubit Allocation and Deallocation
//===----------------------------------------------------------------------===//

/// @brief Allocate a single qubit.
Qubit *__quantum__rt__qubit_allocate();

/// @brief Allocate an array of qubits.
Array *__quantum__rt__qubit_allocate_array(std::uint64_t numQubits);

/// @brief Release a single qubit.
void __quantum__rt__qubit_release(Qubit *q);

/// @brief Release an array of qubits.
void __quantum__rt__qubit_release_array(Array *arr);

/// @brief Deallocate all qubits specified by their indices.
void __quantum__rt__deallocate_all(std::size_t numQubits,
                                   const std::size_t *qubitIdxs);

//===----------------------------------------------------------------------===//
// Single-Qubit Gates (No Parameters)
//===----------------------------------------------------------------------===//

// Hadamard gate
void __quantum__qis__h(Qubit *qubit);

// Rx rotation gate
void __quantum__qis__rx(double param, Qubit *qubit);

//===----------------------------------------------------------------------===//
// Two-Qubit Gates
//===----------------------------------------------------------------------===//

// CNOT (controlled-X) gate
void __quantum__qis__cnot(Qubit *q, Qubit *r);

//===----------------------------------------------------------------------===//
// Measurement Operations
//===----------------------------------------------------------------------===//

/// @brief Measure a qubit in the Z basis.
Result *__quantum__qis__mz(Qubit *q);

/// @brief Measure a qubit and record to a named register.
Result *__quantum__qis__mz__to__register(Qubit *q, const char *name);

//===----------------------------------------------------------------------===//
// Reset Operations
//===----------------------------------------------------------------------===//

/// @brief Reset a qubit to the |0⟩ state.
void __quantum__qis__reset(Qubit *q);

/// @brief Record a measurement result to output.
void __quantum__rt__result_record_output(Result *r, int8_t *name);

} // extern "C"
