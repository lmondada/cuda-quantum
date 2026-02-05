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

/// @brief Construct a `qvector` with `size` qudits in the |0> state.
template <std::size_t Levels>
qvector<Levels>::qvector(std::size_t size) : qudits(size) {}

/// @brief Nullary constructor
/// meant to be used with `kernel_builder<cudaq::qvector<>>`
template <std::size_t Levels>
qvector<Levels>::qvector() : qudits(1) {}

/// @brief Construct a `qvector` from an input state vector.
/// The number of qubits is determined by the size of the input vector.
/// If `validate` is set, it will check the norm of input state vector.
template <std::size_t Levels>
qvector<Levels>::qvector(const std::vector<complex> &vector, bool validate)
    : qudits(std::log2(vector.size())) {
  if (Levels == 2) {
    if (vector.empty() || (vector.size() & (vector.size() - 1)) != 0)
      throw std::runtime_error(
          "Invalid state vector passed to qvector initialization - number of "
          "elements must be power of 2.");
  }
  if (validate) {
    auto norm = std::inner_product(
                    vector.begin(), vector.end(), vector.begin(),
                    complex{0., 0.}, [](auto a, auto b) { return a + b; },
                    [](auto a, auto b) { return std::conj(a) * b; })
                    .real();
    if (std::fabs(1.0 - norm) > 1e-4)
      throw std::runtime_error("Invalid vector norm for qudit allocation.");
  }
  std::vector<QuditInfo> targets;
  for (auto &q : qudits)
    targets.emplace_back(QuditInfo{Levels, q.id()});

  auto precision = std::is_same_v<complex::value_type, float>
                       ? simulation_precision::fp32
                       : simulation_precision::fp64;
  getExecutionManager()->initializeState(targets, vector.data(), precision);
}

template <std::size_t Levels>
qvector<Levels>::qvector(const std::vector<complex> &vector)
    : qvector(vector, /*validate=*/false) {}

template <std::size_t Levels>
qvector<Levels>::qvector(const std::vector<double> &vector)
    : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}

template <std::size_t Levels>
qvector<Levels>::qvector(std::vector<double> &&vector)
    : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}

template <std::size_t Levels>
qvector<Levels>::qvector(const std::initializer_list<double> &list)
    : qvector(std::vector<complex>{list.begin(), list.end()}) {}

template <std::size_t Levels>
qvector<Levels>::qvector(const std::vector<float> &vector)
    : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}

template <std::size_t Levels>
qvector<Levels>::qvector(std::vector<float> &&vector)
    : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}

template <std::size_t Levels>
qvector<Levels>::qvector(const std::initializer_list<float> &list)
    : qvector(std::vector<complex>{list.begin(), list.end()}) {}

template <std::size_t Levels>
qvector<Levels>::qvector(const std::initializer_list<complex> &list)
    : qvector(std::vector<complex>{list.begin(), list.end()}) {}

/// @brief Construct a `qvector` from a pre-existing `state`.
/// This `state` could be constructed with `state::from_data` or retrieved
/// from an cudaq::get_state.
template <std::size_t Levels>
qvector<Levels>::qvector(const cudaq::state &state)
    : qudits(state.get_num_qubits()) {
  std::vector<QuditInfo> targets;
  for (auto &q : qudits)
    targets.emplace_back(QuditInfo{Levels, q.id()});
  // Note: the internal state data will be cloned by the simulator backend.
  getExecutionManager()->initializeState(targets, state.internal.get());
}

template <std::size_t Levels>
qvector<Levels>::qvector(const cudaq::state *ptr) : qvector(*ptr) {}

template <std::size_t Levels>
qvector<Levels>::qvector(cudaq::state *ptr) : qvector(*ptr) {}

template <std::size_t Levels>
qvector<Levels>::qvector(cudaq::state &s)
    : qvector(const_cast<const cudaq::state &>(s)) {}

} // namespace cudaq

#else
static_assert(false,
              "This header should only be included in library mode. "
              "It provides the implementation of qvector methods that interact "
              "with the ExecutionManager.");
#endif
