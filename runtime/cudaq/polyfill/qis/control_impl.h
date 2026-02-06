/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Provide implementations for kernel control flow operations that can be
// emulated in library mode. Automatically included from the cudaq/qis headers
// when in library mode.
//
// Do not include this header directly.

#pragma once

#ifdef CUDAQ_LIBRARY_MODE

namespace cudaq {

//===----------------------------------------------------------------------===//
// control implementations
//===----------------------------------------------------------------------===//

// Control the given cudaq kernel on the given control qubit
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
void control(QuantumKernel &&kernel, qubit &control, Args &&...args) {
  std::vector<std::size_t> ctrls{control.id()};
  getExecutionManager()->startCtrlRegion(ctrls);
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endCtrlRegion(ctrls.size());
}

// Control the given cudaq kernel on the given register of control qubits
template <typename QuantumKernel, typename QuantumRegister, typename... Args>
  requires std::ranges::range<QuantumRegister> &&
           isCallableVoidKernel<QuantumKernel, Args...>
void control(QuantumKernel &&kernel, QuantumRegister &&ctrl_qubits,
             Args &&...args) {
  std::vector<std::size_t> ctrls;
  for (std::size_t i = 0; i < ctrl_qubits.size(); i++) {
    ctrls.push_back(ctrl_qubits[i].id());
  }
  getExecutionManager()->startCtrlRegion(ctrls);
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endCtrlRegion(ctrls.size());
}

// Control the given cudaq kernel on the given list of references to control
// qubits.
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
void control(QuantumKernel &&kernel,
             std::vector<std::reference_wrapper<qubit>> &&ctrl_qubits,
             Args &&...args) {
  std::vector<std::size_t> ctrls;
  for (auto &cq : ctrl_qubits) {
    ctrls.push_back(cq.get().id());
  }
  getExecutionManager()->startCtrlRegion(ctrls);
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endCtrlRegion(ctrls.size());
}

//===----------------------------------------------------------------------===//
// adjoint implementation
//===----------------------------------------------------------------------===//

// Apply the adjoint of the given cudaq kernel
template <typename QuantumKernel, typename... Args>
  requires isCallableVoidKernel<QuantumKernel, Args...>
void adjoint(QuantumKernel &&kernel, Args &&...args) {
  // static_assert(true, "adj not implemented yet.");
  getExecutionManager()->startAdjointRegion();
  kernel(std::forward<Args>(args)...);
  getExecutionManager()->endAdjointRegion();
}

//===----------------------------------------------------------------------===//
// compute_action / compute_dag_action implementations
//===----------------------------------------------------------------------===//

/// Instantiate this type to affect C A C^dag, where the user
/// provides cudaq Kernels C and A (compute, action).
// struct compute_action {
template <typename ComputeFunction, typename ActionFunction>
  requires isCallableVoidKernel<ComputeFunction> &&
           isCallableVoidKernel<ActionFunction>
void compute_action(ComputeFunction &&c, ActionFunction &&a) {
  c();
  a();
  adjoint(c);
}

/// Instantiate this type to affect C^dag A C, where the user
/// provides cudaq Kernels C and A (compute, action).
// struct compute_dag_action {
template <typename ComputeFunction, typename ActionFunction>
  requires isCallableVoidKernel<ComputeFunction> &&
           isCallableVoidKernel<ActionFunction>
void compute_dag_action(ComputeFunction &&c, ActionFunction &&a) {
  adjoint(c);
  a();
  c();
}

} // namespace cudaq

#else
static_assert(false,
              "This header should only be included in library mode. "
              "It provides the implementation of kernel control flow functions "
              "that interact with the ExecutionManager.");
#endif
