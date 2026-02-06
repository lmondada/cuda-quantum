/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Provide implementations for QPU runtime features that can be emulated in
// library mode. Automatically included from the cudaq/qis headers when in
// library mode.
//
// Do not include this header directly.

#pragma once

#ifdef CUDAQ_LIBRARY_MODE

namespace cudaq {

//===----------------------------------------------------------------------===//
// Helper function implementations
//===----------------------------------------------------------------------===//

/// This function will apply the specified `QuantumOp`. It will check the
/// modifier template type and if it is `base`, it will apply the operation to
/// any qubits provided as input. If `ctrl`, it will take the first `N-1` qubits
/// as the controls and the last qubit as the target.
template <typename QuantumOp, typename mod, typename... QubitArgs>
void oneQubitApply(QubitArgs &...args) {
  // Get the name of this operation
  auto gateName = QuantumOp::name();
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");

  // Get the number of input qubits
  constexpr std::size_t nArgs = sizeof...(QubitArgs);

  // Map the qubits to their unique ids and pack them.
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(args)...};
  std::vector<bool> qubitIsNegated{qubitIsNegative(args)...};

  // If mod == base, then we just want to apply the gate to all qubits provided.
  // This is a broadcast application.
  if constexpr (std::is_same_v<mod, base>) {
    for (auto &qubit : quditInfos)
      getExecutionManager()->apply(gateName, {}, {}, {qubit});

    // Nothing left to do, return
    return;
  }

  // If we are here, then `mod` must be control or adjoint. Extract the controls
  // and the target
  std::vector<QuditInfo> controls(quditInfos.begin(),
                                  quditInfos.begin() + nArgs - 1);

  // If we have controls, check if any of them are negative controls, and if so
  // apply an x.
  if (!controls.empty())
    for (std::size_t i = 0; i < controls.size(); i++)
      if (qubitIsNegated[i])
        getExecutionManager()->apply("x", {}, {}, {controls[i]});

  // Apply the gate
  getExecutionManager()->apply(gateName, {}, controls, {quditInfos.back()},
                               std::is_same_v<mod, adj>);

  // If we did apply any X ops for a negative control, we need to reverse it.
  if (!controls.empty()) {
    for (std::size_t i = 0; i < controls.size(); i++) {
      if (qubitIsNegated[i]) {
        getExecutionManager()->apply("x", {}, {}, {controls[i]});
        // fold expression which will reverse the negation
        (
            [&] {
              if (args.is_negative())
                args.negate();
            }(),
            ...);
      }
    }
  }
}

/// This function will apply a multi-controlled operation with the given control
/// register on the single qubit target.
template <typename QuantumOp, typename mod, typename QubitRange>
  requires(std::ranges::range<QubitRange>)
void oneQubitApplyControlledRange(QubitRange &ctrls, qubit &target) {
  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the input control register to a vector of QuditInfo
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](auto &q) { return cudaq::qubitToQuditInfo(q); });

  // Apply the gate
  getExecutionManager()->apply(gateName, {}, controls,
                               {cudaq::qubitToQuditInfo(target)});
}

template <typename QuantumOp, typename mod, typename ScalarAngle,
          typename... QubitArgs>
void oneQubitSingleParameterApply(ScalarAngle angle, QubitArgs &...args) {
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");
  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the qubits to their unique ids and pack them into a std::array
  constexpr std::size_t nArgs = sizeof...(QubitArgs);
  std::vector<QuditInfo> targets{qubitToQuditInfo(args)...};
  std::vector<double> parameters{static_cast<double>(angle)};

  // If there are more than one qubits and mod == base, then
  // we just want to apply the same gate to all qubits provided
  if constexpr (nArgs > 1 && std::is_same_v<mod, base>) {
    for (auto &targetId : targets)
      getExecutionManager()->apply(gateName, parameters, {}, {targetId});

    // Nothing left to do, return
    return;
  }

  // If we are here, then mod must be control or adjoint
  // Extract the controls and the target
  std::vector<QuditInfo> controls(targets.begin(), targets.begin() + nArgs - 1);

  // Apply the gate
  getExecutionManager()->apply(gateName, parameters, controls, {targets.back()},
                               std::is_same_v<mod, adj>);
}

template <typename QuantumOp, typename mod, typename ScalarAngle,
          typename QubitRange>
  requires(std::ranges::range<QubitRange>)
void oneQubitSingleParameterControlledRange(ScalarAngle angle,
                                            QubitRange &ctrls, qubit &target) {
  // Get the name of the operation
  auto gateName = QuantumOp::name();

  // Map the input control register to a vector of QuditInfo
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return qubitToQuditInfo(q); });

  // Apply the gate
  getExecutionManager()->apply(gateName, {angle}, controls,
                               {qubitToQuditInfo(target)});
}

//===----------------------------------------------------------------------===//
// Gate macro implementation version
//===----------------------------------------------------------------------===//

#define CUDAQ_QIS_ONE_TARGET_QUBIT_IMPL_(NAME)                                 \
  template <typename mod, typename... QubitArgs>                               \
  void NAME(QubitArgs &...args) {                                              \
    oneQubitApply<qubit_op::NAME##Op, mod>(args...);                           \
  }                                                                            \
  template <typename mod, typename QubitRange>                                 \
    requires(std::ranges::range<QubitRange>)                                   \
  void NAME(QubitRange &ctrls, qubit &target) {                                \
    oneQubitApplyControlledRange<qubit_op::NAME##Op, mod>(ctrls, target);      \
  }                                                                            \
  template <typename mod, typename QubitRange>                                 \
    requires(std::ranges::range<QubitRange>)                                   \
  void NAME(QubitRange &qr) {                                                  \
    for (auto &q : qr) {                                                       \
      NAME<mod>(q);                                                            \
    }                                                                          \
  }                                                                            \
  template <typename mod, typename QubitRange>                                 \
    requires(std::ranges::range<QubitRange>)                                   \
  void NAME(QubitRange &&qr) {                                                 \
    for (auto &q : qr) {                                                       \
      NAME<mod>(q);                                                            \
    }                                                                          \
  }

// Instantiate implementations for the default logical gate set
CUDAQ_QIS_ONE_TARGET_QUBIT_IMPL_(h)
CUDAQ_QIS_ONE_TARGET_QUBIT_IMPL_(x)
CUDAQ_QIS_ONE_TARGET_QUBIT_IMPL_(y)
CUDAQ_QIS_ONE_TARGET_QUBIT_IMPL_(z)
CUDAQ_QIS_ONE_TARGET_QUBIT_IMPL_(t)
CUDAQ_QIS_ONE_TARGET_QUBIT_IMPL_(s)

#undef CUDAQ_QIS_ONE_TARGET_QUBIT_IMPL_

#define CUDAQ_QIS_PARAM_ONE_TARGET_IMPL_(NAME)                                 \
  template <typename mod, typename ScalarAngle, typename... QubitArgs>         \
  void NAME(ScalarAngle angle, QubitArgs &...args) {                           \
    oneQubitSingleParameterApply<qubit_op::NAME##Op, mod>(angle, args...);     \
  }                                                                            \
  template <typename mod, typename ScalarAngle, typename QubitRange>           \
    requires(std::ranges::range<QubitRange>)                                   \
  void NAME(ScalarAngle angle, QubitRange &ctrls, qubit &target) {             \
    oneQubitSingleParameterControlledRange<qubit_op::NAME##Op, mod>(           \
        angle, ctrls, target);                                                 \
  }

CUDAQ_QIS_PARAM_ONE_TARGET_IMPL_(rx)
CUDAQ_QIS_PARAM_ONE_TARGET_IMPL_(ry)
CUDAQ_QIS_PARAM_ONE_TARGET_IMPL_(rz)
CUDAQ_QIS_PARAM_ONE_TARGET_IMPL_(r1)

#undef CUDAQ_QIS_PARAM_ONE_TARGET_IMPL_

//===----------------------------------------------------------------------===//
// u3 gate implementation
//===----------------------------------------------------------------------===//

template <typename mod, typename ScalarAngle, typename... QubitArgs>
void u3(ScalarAngle theta, ScalarAngle phi, ScalarAngle lambda,
        QubitArgs &...args) {
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");

  std::vector<ScalarAngle> parameters{theta, phi, lambda};

  // Map the qubits to their unique ids and pack them into a std::array
  constexpr std::size_t nArgs = sizeof...(QubitArgs);
  std::vector<QuditInfo> targets{qubitToQuditInfo(args)...};

  // If there are more than one qubits and mod == base, then
  // we just want to apply the same gate to all qubits provided
  if constexpr (nArgs > 1 && std::is_same_v<mod, base>) {
    for (auto &targetId : targets)
      getExecutionManager()->apply("u3", parameters, {}, {targetId});
    return;
  }

  // If we are here, then mod must be control or adjoint
  // Extract the controls and the target
  std::vector<QuditInfo> controls(targets.begin(), targets.begin() + nArgs - 1);

  // Apply the gate
  getExecutionManager()->apply("u3", parameters, controls, {targets.back()},
                               std::is_same_v<mod, adj>);
}

template <typename mod, typename ScalarAngle, typename QubitRange>
  requires(std::ranges::range<QubitRange>)
void u3(ScalarAngle theta, ScalarAngle phi, ScalarAngle lambda,
        QubitRange &ctrls, qubit &target) {
  std::vector<ScalarAngle> parameters{theta, phi, lambda};
  // Map the input control register to a vector of QuditInfo
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return qubitToQuditInfo(q); });

  // Apply the gate
  getExecutionManager()->apply("u3", parameters, controls,
                               {qubitToQuditInfo(target)});
}

//===----------------------------------------------------------------------===//
// swap gate implementation
//===----------------------------------------------------------------------===//

template <typename mod, typename... QubitArgs>
void swap(QubitArgs &...args) {
  static_assert(std::conjunction<std::is_same<qubit, QubitArgs>...>::value,
                "Cannot operate on a qudit with Levels != 2");
  constexpr std::size_t nArgs = sizeof...(QubitArgs);
  std::vector<QuditInfo> qubitIds{qubitToQuditInfo(args)...};
  if constexpr (nArgs == 2) {
    getExecutionManager()->apply("swap", {}, {}, qubitIds);
    return;
  } else {
    static_assert(std::is_same_v<mod, ctrl>,
                  "More than 2 qubits passed to swap but modifier != ctrl.");
  }

  // Controls are all qubits except the last 2
  std::vector<QuditInfo> controls(qubitIds.begin(),
                                  qubitIds.begin() + qubitIds.size() - 2);
  std::vector<QuditInfo> targets(qubitIds.end() - 2, qubitIds.end());
  getExecutionManager()->apply("swap", {}, controls, targets);
}

template <typename QuantumRegister>
  requires(std::ranges::range<QuantumRegister>)
void swap(QuantumRegister &ctrls, qubit &src, qubit &target) {
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return qubitToQuditInfo(q); });
  getExecutionManager()->apply(
      "swap", {}, controls, {qubitToQuditInfo(src), qubitToQuditInfo(target)});
}

//===----------------------------------------------------------------------===//
// Common 2-qubit gate implementations
//===----------------------------------------------------------------------===//

inline void cnot(qubit &q, qubit &r) { x<cudaq::ctrl>(q, r); }
inline void cx(qubit &q, qubit &r) { x<cudaq::ctrl>(q, r); }
inline void cy(qubit &q, qubit &r) { y<cudaq::ctrl>(q, r); }
inline void cz(qubit &q, qubit &r) { z<cudaq::ctrl>(q, r); }
inline void ch(qubit &q, qubit &r) { h<cudaq::ctrl>(q, r); }
inline void cs(qubit &q, qubit &r) { s<cudaq::ctrl>(q, r); }
inline void ct(qubit &q, qubit &r) { t<cudaq::ctrl>(q, r); }
inline void ccx(qubit &q, qubit &r, qubit &s) { x<cudaq::ctrl>(q, r, s); }

// Define common 2 qubit operations with angles.
template <typename T>
void crx(T angle, qubit &q, qubit &r) {
  rx<cudaq::ctrl>(angle, q, r);
}
template <typename T>
void cry(T angle, qubit &q, qubit &r) {
  ry<cudaq::ctrl>(angle, q, r);
}
template <typename T>
void crz(T angle, qubit &q, qubit &r) {
  rz<cudaq::ctrl>(angle, q, r);
}
template <typename T>
void cr1(T angle, qubit &q, qubit &r) {
  r1<cudaq::ctrl>(angle, q, r);
}

// Define common single qubit adjoint operations.
inline void sdg(qubit &q) { s<cudaq::adj>(q); }
inline void tdg(qubit &q) { t<cudaq::adj>(q); }

//===----------------------------------------------------------------------===//
// exp_pauli implementations
//===----------------------------------------------------------------------===//

/// @brief Apply a general Pauli rotation, takes a qubit register and the size
/// must be equal to the Pauli word length.
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
void exp_pauli(double theta, QubitRange &&qubits, const char *pauliWord) {
  std::vector<QuditInfo> quditInfos;
  std::transform(qubits.begin(), qubits.end(), std::back_inserter(quditInfos),
                 [](auto &q) { return cudaq::qubitToQuditInfo(q); });
  // FIXME: it would be cleaner if we just kept it as a pauli word here
  getExecutionManager()->apply("exp_pauli", {theta}, {}, quditInfos, false,
                               spin_op::from_word(pauliWord));
}

/// @brief Apply a general Pauli rotation, takes a qubit register and the size
/// must be equal to the Pauli word length.
template <typename QubitRange>
  requires(std::ranges::range<QubitRange>)
void exp_pauli(double theta, QubitRange &&qubits,
               const cudaq::pauli_word &pauliWord) {
  exp_pauli(theta, qubits, pauliWord.str().c_str());
}

/// @brief Apply a general Pauli rotation, takes a variadic set of
/// qubits, and the number of qubits must be equal to the Pauli word length.
template <typename... QubitArgs>
void exp_pauli(double theta, const char *pauliWord, QubitArgs &...qubits) {

  if (sizeof...(QubitArgs) != std::strlen(pauliWord))
    throw std::runtime_error(
        "Invalid exp_pauli call, number of qubits != size of pauliWord.");

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(qubits)...};
  getExecutionManager()->apply("exp_pauli", {theta}, {}, quditInfos, false,
                               spin_op::from_word(pauliWord));
}

/// @brief Apply a general Pauli rotation with control qubits and a variadic set
/// of qubits. The number of qubits must be equal to the Pauli word length.
template <typename QuantumRegister, typename... QubitArgs>
  requires(std::ranges::range<QuantumRegister>)
void exp_pauli(QuantumRegister &ctrls, double theta, const char *pauliWord,
               QubitArgs &...qubits) {
  std::vector<QuditInfo> controls;
  std::transform(ctrls.begin(), ctrls.end(), std::back_inserter(controls),
                 [](const auto &q) { return qubitToQuditInfo(q); });
  if (sizeof...(QubitArgs) != std::strlen(pauliWord))
    throw std::runtime_error(
        "Invalid exp_pauli call, number of qubits != size of pauliWord.");

  // Map the qubits to their unique ids and pack them into a std::array
  std::vector<QuditInfo> quditInfos{qubitToQuditInfo(qubits)...};
  getExecutionManager()->apply("exp_pauli", {theta}, controls, quditInfos,
                               false, spin_op::from_word(pauliWord));
}

} // namespace cudaq

#else
static_assert(false,
              "This header should only be included in library mode. "
              "It provides the implementation of gate functions that interact "
              "with the ExecutionManager.");
#endif
