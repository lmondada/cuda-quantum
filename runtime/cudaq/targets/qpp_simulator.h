/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
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

/// @brief Allocate qubits initialized with single-precision floating-point
/// state amplitudes.
Array *
__quantum__rt__qubit_allocate_array_with_state_fp32(std::uint64_t numQubits,
                                                    const float *data);

/// @brief Allocate qubits initialized with double-precision floating-point
/// state amplitudes.
Array *
__quantum__rt__qubit_allocate_array_with_state_fp64(std::uint64_t numQubits,
                                                    const double *data);

/// @brief Allocate qubits initialized with single-precision complex state
/// amplitudes.
Array *__quantum__rt__qubit_allocate_array_with_state_complex32(
    std::uint64_t numQubits, const std::complex<float> *data);

/// @brief Allocate qubits initialized with double-precision complex state
/// amplitudes.
Array *__quantum__rt__qubit_allocate_array_with_state_complex64(
    std::uint64_t numQubits, const std::complex<double> *data);

/// @brief Allocate qubits initialized with a SimulationState pointer.
Array *__quantum__rt__qubit_allocate_array_with_state_ptr(
    cudaq::SimulationState *state);

/// @brief Allocate qubits initialized with a cudaq::state pointer.
Array *__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(
    std::uint64_t numQubits, cudaq::state *state);

/// @brief Release a single qubit.
void __quantum__rt__qubit_release(Qubit *q);

/// @brief Release an array of qubits.
void __quantum__rt__qubit_release_array(Array *arr);

/// @brief Deallocate all qubits specified by their indices.
void __quantum__rt__deallocate_all(std::size_t numQubits,
                                   const std::size_t *qubitIdxs);

//===----------------------------------------------------------------------===//
// Array Operations
//===----------------------------------------------------------------------===//

/// @brief Create a 1D array with specified item size and count.
Array *__quantum__rt__array_create_1d(int32_t itemSizeInBytes,
                                      int64_t count_items);

/// @brief Release an array.
void __quantum__rt__array_release(Array *a);

/// @brief Get the size of a 1D array.
int64_t __quantum__rt__array_get_size_1d(Array *array);

/// @brief Get a pointer to an element in a 1D array.
int8_t *__quantum__rt__array_get_element_ptr_1d(Array *q, uint64_t idx);

/// @brief Copy an array, optionally forcing a new instance.
Array *__quantum__rt__array_copy(Array *array, bool forceNewInstance);

/// @brief Concatenate two arrays.
Array *__quantum__rt__array_concatenate(Array *head, Array *tail);

/// @brief Slice an array along a specified dimension.
Array *__quantum__rt__array_slice(Array *array, int32_t dim,
                                  int64_t range_start, int64_t range_step,
                                  int64_t range_end);

/// @brief Slice a 1D array.
Array *__quantum__rt__array_slice_1d(Array *array, int64_t range_start,
                                     int64_t range_step, int64_t range_end);

//===----------------------------------------------------------------------===//
// Single-Qubit Gates (No Parameters)
//===----------------------------------------------------------------------===//

// Hadamard gate
void __quantum__qis__h(Qubit *qubit);
void __quantum__qis__h__ctl(Array *ctrlQubits, Qubit *qubit);

// Pauli-X gate
void __quantum__qis__x(Qubit *qubit);
void __quantum__qis__x__ctl(Array *ctrlQubits, Qubit *qubit);

// Pauli-Y gate
void __quantum__qis__y(Qubit *qubit);
void __quantum__qis__y__ctl(Array *ctrlQubits, Qubit *qubit);

// Pauli-Z gate
void __quantum__qis__z(Qubit *qubit);
void __quantum__qis__z__ctl(Array *ctrlQubits, Qubit *qubit);

// T gate
void __quantum__qis__t(Qubit *qubit);
void __quantum__qis__t__ctl(Array *ctrlQubits, Qubit *qubit);
void __quantum__qis__t__adj(Qubit *qubit);

// S gate
void __quantum__qis__s(Qubit *qubit);
void __quantum__qis__s__ctl(Array *ctrlQubits, Qubit *qubit);
void __quantum__qis__s__adj(Qubit *qubit);

// T-dagger gate
void __quantum__qis__tdg(Qubit *qubit);
void __quantum__qis__tdg__ctl(Array *ctrlQubits, Qubit *qubit);

// S-dagger gate
void __quantum__qis__sdg(Qubit *qubit);
void __quantum__qis__sdg__ctl(Array *ctrlQubits, Qubit *qubit);

//===----------------------------------------------------------------------===//
// Single-Qubit Rotation Gates (One Parameter)
//===----------------------------------------------------------------------===//

// Rx rotation gate
void __quantum__qis__rx(double param, Qubit *qubit);
void __quantum__qis__rx__ctl(double param, Array *ctrlQubits, Qubit *qubit);

// Ry rotation gate
void __quantum__qis__ry(double param, Qubit *qubit);
void __quantum__qis__ry__ctl(double param, Array *ctrlQubits, Qubit *qubit);

// Rz rotation gate
void __quantum__qis__rz(double param, Qubit *qubit);
void __quantum__qis__rz__ctl(double param, Array *ctrlQubits, Qubit *qubit);

// R1 (phase) rotation gate
void __quantum__qis__r1(double param, Qubit *qubit);
void __quantum__qis__r1__ctl(double param, Array *ctrlQubits, Qubit *qubit);

// Phased-Rx gate (two parameters)
void __quantum__qis__phased_rx(double theta, double phi, Qubit *q);

//===----------------------------------------------------------------------===//
// Two-Qubit Gates
//===----------------------------------------------------------------------===//

// CNOT (controlled-X) gate
void __quantum__qis__cnot(Qubit *q, Qubit *r);

// Controlled-Z gate
void __quantum__qis__cz(Qubit *q, Qubit *r);

// SWAP gate
void __quantum__qis__swap(Qubit *q, Qubit *r);
void __quantum__qis__swap__ctl(Array *ctrls, Qubit *q, Qubit *r);

// Controlled phase gate
void __quantum__qis__cphase(double d, Qubit *q, Qubit *r);

//===----------------------------------------------------------------------===//
// U3 Gate (Three Parameters)
//===----------------------------------------------------------------------===//

/// @brief General single-qubit rotation with three Euler angles.
void __quantum__qis__u3(double theta, double phi, double lambda, Qubit *q);

/// @brief Controlled U3 gate.
void __quantum__qis__u3__ctl(double theta, double phi, double lambda,
                             Array *ctrls, Qubit *q);

//===----------------------------------------------------------------------===//
// Measurement Operations
//===----------------------------------------------------------------------===//

/// @brief Measure a qubit in the Z basis.
Result *__quantum__qis__mz(Qubit *q);

/// @brief Measure a qubit and record to a named register.
Result *__quantum__qis__mz__to__register(Qubit *q, const char *name);

/// @brief Read the boolean value of a measurement result (QIR 1.0).
bool __quantum__rt__read_result(Result *result);

//===----------------------------------------------------------------------===//
// Reset Operations
//===----------------------------------------------------------------------===//

/// @brief Reset a qubit to the |0⟩ state.
void __quantum__qis__reset(Qubit *q);

//===----------------------------------------------------------------------===//
// Custom Unitary Operations
//===----------------------------------------------------------------------===//

/// @brief Apply a custom unitary matrix to target qubits.
void __quantum__qis__custom_unitary(std::complex<double> *unitary,
                                    Array *controls, Array *targets,
                                    const char *name);

/// @brief Apply the adjoint of a custom unitary matrix to target qubits.
void __quantum__qis__custom_unitary__adj(std::complex<double> *unitary,
                                         Array *controls, Array *targets,
                                         const char *name);

//===----------------------------------------------------------------------===//
// Exponential Pauli Operations
//===----------------------------------------------------------------------===//

/// @brief Apply exp(i * theta * P) where P is a Pauli word.
void __quantum__qis__exp_pauli(double theta, Array *qubits, char *pauliWord);
void __quantum__qis__exp_pauli__ctl(double theta, Array *ctrls, Array *qubits,
                                    char *pauliWord);

//===----------------------------------------------------------------------===//
// Noise and Error Operations
//===----------------------------------------------------------------------===//

/// @brief Apply a generalized Kraus channel (noise operation).
/// @param dataKind 0 for float, 1 for double.
void __quantum__qis__apply_kraus_channel_generalized(
    std::int64_t dataKind, std::int64_t krausChannelKey, std::size_t numSpans,
    std::size_t numParams, std::size_t numTargets, ...);

/// @brief Trigger a runtime trap with an error code.
void __quantum__qis__trap(std::int64_t code);

//===----------------------------------------------------------------------===//
// Output Recording Functions
//===----------------------------------------------------------------------===//

/// @brief Record a measurement result to output.
void __quantum__rt__result_record_output(Result *r, int8_t *name);

/// @brief Record a boolean value to output.
void __quantum__rt__bool_record_output(bool val, const char *label);

/// @brief Record an integer value to output.
void __quantum__rt__int_record_output(std::int64_t val, const char *label);

/// @brief Record a double value to output.
void __quantum__rt__double_record_output(double val, const char *label);

/// @brief Record a tuple structure to output.
void __quantum__rt__tuple_record_output(std::uint64_t len, const char *label);

/// @brief Record an array structure to output.
void __quantum__rt__array_record_output(std::uint64_t len, const char *label);

/// @brief Clear result maps between consecutive programs.
void __quantum__rt__clear_result_maps();

//===----------------------------------------------------------------------===//
// Array/Vector Conversion Utilities
//===----------------------------------------------------------------------===//

/// @brief Convert a QIR Array to a std::vector for C++ interop.
void *__quantum__qis__convert_array_to_stdvector(Array *arr);

/// @brief Free a converted std::vector.
void __quantum__qis__free_converted_stdvector(void *veq);

//===----------------------------------------------------------------------===//
// Control Qubit Invocation Helpers
//===----------------------------------------------------------------------===//

/// @brief Invoke a QIS function with a variadic list of control qubits.
void invokeWithControlQubits(std::size_t numControlOperands,
                             void (*QISFunction)(Array *, Qubit *), ...);

/// @brief Invoke a controlled rotation with control qubits.
void invokeRotationWithControlQubits(
    double param, std::size_t numControlOperands, std::size_t *isArrayAndLength,
    void (*QISFunction)(double, Array *, Qubit *), ...);

/// @brief Invoke a controlled U3 rotation with control qubits.
void invokeU3RotationWithControlQubits(
    double theta, double phi, double lambda, std::size_t numControlOperands,
    std::size_t *isArrayAndLength,
    void (*QISFunction)(double, double, double, Array *, Qubit *), ...);

/// @brief Invoke a QIS function with control registers or qubits.
void invokeWithControlRegisterOrQubits(std::size_t numControlOperands,
                                       std::size_t *isArrayAndLength,
                                       std::size_t numTargetOperands,
                                       void (*QISFunction)(Array *, Qubit *),
                                       ...);

/// @brief Generalized invoke with rotations, controls, and targets.
void generalizedInvokeWithRotationsControlsTargets(
    std::size_t numRotationOperands, std::size_t numControlArrayOperands,
    std::size_t numControlQubitOperands, std::size_t numTargetOperands,
    void (*QISFunction)(...), ...);

//===----------------------------------------------------------------------===//
// Debug/Utility Functions
//===----------------------------------------------------------------------===//

/// @brief Print an integer value with a format string.
void print_i64(const char *msg, std::size_t i);

/// @brief Print a double value with a format string.
void print_f64(const char *msg, double f);

} // extern "C"
