/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"
#include "cudaq/qis/qview.h"
#include "cudaq/qis/state.h"

namespace cudaq {

/// @brief A `qvector` is an owning, dynamically sized container for qudits.
/// The semantics of the `qvector` follows that of a `std::vector` for qudits.
/// It is templated on the number of levels for the held qudits.
template <std::size_t Levels = 2>
class qvector {
public:
  /// @brief Useful typedef indicating the underlying qudit type
  using value_type = qudit<Levels>;

private:
  /// @brief Reference to the held / owned vector of qudits.
  std::vector<value_type> qudits;

public:
  /// @brief Construct a `qvector` with `size` qudits in the |0> state.
  qvector(std::size_t size);

  /// @cond
  /// Nullary constructor
  /// meant to be used with `kernel_builder<cudaq::qvector<>>`
  qvector();
  /// @endcond

  /// @brief Construct a `qvector` from an input state vector.
  /// The number of qubits is determined by the size of the input vector.
  /// If `validate` is set, it will check the norm of input state vector.
  explicit qvector(const std::vector<complex> &vector, bool validate);
  qvector(const std::vector<complex> &vector);

  qvector(const std::vector<double> &vector);
  qvector(std::vector<double> &&vector);
  qvector(const std::initializer_list<double> &list);
  qvector(const std::vector<float> &vector);
  qvector(std::vector<float> &&vector);
  qvector(const std::initializer_list<float> &list);
  qvector(const std::initializer_list<complex> &list);

  //===--------------------------------------------------------------------===//
  // qvector with an initial state
  //===--------------------------------------------------------------------===//
  /// @brief Construct a `qvector` from a pre-existing `state`.
  /// This `state` could be constructed with `state::from_data` or retrieved
  /// from an cudaq::get_state.
  qvector(const cudaq::state &state);
  qvector(const cudaq::state *ptr);
  qvector(cudaq::state *ptr);
  qvector(cudaq::state &s);

  /// @brief `qvectors` cannot be copied
  qvector(qvector const &) = delete;

  /// @brief `qvectors` cannot be moved
  qvector(qvector &&) = delete;

  /// @brief `qvectors` cannot be copy assigned.
  qvector &operator=(const qvector &) = delete;

  /// @brief Iterator interface, begin.
  auto begin() { return qudits.begin(); }

  /// @brief Iterator interface, end.
  auto end() { return qudits.end(); }

  /// @brief Returns the qudit at `idx`.
  value_type &operator[](const std::size_t idx) { return qudits[idx]; }

  /// @brief Returns the `[0, count)` qudits as a non-owning `qview`.
  qview<Levels> front(std::size_t count) {
    return std::span(qudits).subspan(0, count);
  }

  /// @brief Returns the first qudit.
  value_type &front() { return qudits.front(); }

  /// @brief Returns the `[count, size())` qudits as a non-owning `qview`
  qview<Levels> back(std::size_t count) {
    return std::span(qudits).subspan(size() - count, count);
  }

  /// @brief Returns the last qudit.
  value_type &back() { return qudits.back(); }

  /// @brief Returns the `[start, start+size)` qudits as a non-owning `qview`
  qview<Levels> slice(std::size_t start, std::size_t size) {
    return std::span(qudits).subspan(start, size);
  }

  /// @brief Returns the number of contained qudits.
  std::size_t size() const { return qudits.size(); }

  /// @brief Destroys all contained qudits. Postcondition: `size() == 0`.
  void clear() { qudits.clear(); }
};

} // namespace cudaq

#ifdef CUDAQ_LIBRARY_MODE
#include "cudaq/polyfill/qis/qvector_impl.h"
#endif
