/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#ifdef CUDAQ_LIBRARY_MODE

namespace cudaq {

/// Construct a qudit, will allocate a new unique index
template <std::size_t Levels>
qudit<Levels>::qudit()
    : idx(getExecutionManager()->allocateQudit(n_levels())) {}

template <std::size_t Levels>
qudit<Levels>::qudit(const std::vector<complex> &state) : qudit() {
  if (state.size() != Levels)
    throw std::runtime_error(
        "Invalid number of state vector elements for qudit allocation (" +
        std::to_string(state.size()) + ").");

  auto norm = std::inner_product(
                  state.begin(), state.end(), state.begin(), complex{0., 0.},
                  [](auto a, auto b) { return a + b; },
                  [](auto a, auto b) { return std::conj(a) * b; })
                  .real();
  if (std::fabs(1.0 - norm) > 1e-4)
    throw std::runtime_error("Invalid vector norm for qudit allocation.");

  // Perform the initialization
  auto precision = std::is_same_v<complex::value_type, float>
                       ? simulation_precision::fp32
                       : simulation_precision::fp64;
  getExecutionManager()->initializeState({QuditInfo(n_levels(), idx)},
                                         state.data(), precision);
}

template <std::size_t Levels>
qudit<Levels>::qudit(const std::initializer_list<complex> &list)
    : qudit({list.begin(), list.end()}) {}

template <std::size_t Levels>
qudit<Levels>::qudit(const cudaq::state &state) : qudit() {
  // Note: the internal state data will be cloned by the simulator backend.
  std::vector<QuditInfo> v{QuditInfo{Levels, id()}};
  getExecutionManager()->initializeState(v, state.internal.get());
}

template <std::size_t Levels>
qudit<Levels>::qudit(const cudaq::state *s) : qudit(*s) {}

template <std::size_t Levels>
qudit<Levels>::qudit(cudaq::state *s)
    : qudit(const_cast<const cudaq::state *>(s)) {}

template <std::size_t Levels>
qudit<Levels>::qudit(cudaq::state &s)
    : qudit(const_cast<const cudaq::state &>(s)) {}

// Destructor, return the qudit so it can be reused
template <std::size_t Levels>
qudit<Levels>::~qudit() {
  getExecutionManager()->returnQudit({n_levels(), idx});
}

} // namespace cudaq

#else
static_assert(false,
              "This header should only be included in library mode. "
              "It provides the implementation of qudit methods that interact "
              "with the ExecutionManager.");
#endif
