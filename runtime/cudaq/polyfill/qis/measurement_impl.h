/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Provide implementations for measurement operations that can be emulated in
// library mode. Automatically included from the cudaq/qis headers when in
// library mode.
//
// Do not include this header directly.

#pragma once

#ifdef CUDAQ_LIBRARY_MODE

namespace cudaq {

//===----------------------------------------------------------------------===//
// Single-qubit measurement implementations
//===----------------------------------------------------------------------===//

/// @brief Measure an individual qubit, return 0,1 as `bool`
inline measure_result mz(qubit &q) {
  return getExecutionManager()->measure(QuditInfo{q.n_levels(), q.id()});
}

/// @brief Measure an individual qubit in `x` basis, return 0,1 as `bool`
inline measure_result mx(qubit &q) {
  h(q);
  return getExecutionManager()->measure(QuditInfo{q.n_levels(), q.id()});
}

// Measure an individual qubit in `y` basis, return 0,1 as `bool`
inline measure_result my(qubit &q) {
  r1(-M_PI_2, q);
  h(q);
  return getExecutionManager()->measure(QuditInfo{q.n_levels(), q.id()});
}

inline void reset(qubit &q) {
  getExecutionManager()->reset({q.n_levels(), q.id()});
}

//===----------------------------------------------------------------------===//
// Multi-qubit measurement implementations
//===----------------------------------------------------------------------===//

// Measure all qubits in the range, return vector of 0,1
template <typename QubitRange>
  requires std::ranges::range<QubitRange>
std::vector<measure_result> mz(QubitRange &q) {
  std::vector<measure_result> b;
  for (auto &qq : q) {
    b.push_back(mz(qq));
  }
  return b;
}

template <std::size_t Levels>
std::vector<measure_result> mz(const qview<Levels> &q) {
  std::vector<measure_result> b;
  for (auto &qq : q) {
    b.emplace_back(mz(qq));
  }
  return b;
}

template <typename QubitRange, typename... Qs>
  requires(std::ranges::range<QubitRange>)
std::vector<measure_result> mz(QubitRange &qr, Qs &&...qs) {
  std::vector<measure_result> result = mz(qr);
  auto rest = mz(std::forward<Qs>(qs)...);
  if constexpr (std::is_same_v<decltype(rest), measure_result>) {
    result.push_back(rest);
  } else {
    result.insert(result.end(), rest.begin(), rest.end());
  }
  return result;
}

template <typename... Qs>
std::vector<measure_result> mz(qubit &q, Qs &&...qs) {
  std::vector<measure_result> result = {mz(q)};
  auto rest = mz(std::forward<Qs>(qs)...);
  if constexpr (std::is_same_v<decltype(rest), measure_result>) {
    result.push_back(rest);
  } else {
    result.insert(result.end(), rest.begin(), rest.end());
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Spin measurement implementation
//===----------------------------------------------------------------------===//

// Measure the state in the given spin_op basis.
inline SpinMeasureResult measure(const cudaq::spin_op &term) {
  return getExecutionManager()->measure(term);
}

} // namespace cudaq

#else
static_assert(false,
              "This header should only be included in library mode. "
              "It provides the implementation of measurement functions that "
              "interact with the ExecutionManager.");
#endif
