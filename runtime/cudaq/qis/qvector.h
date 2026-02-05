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

// Forward declarations
class Array;
struct Qubit;
using Result = bool;

namespace cudaq {

#ifndef CUDAQ_NO_MLIR_MODE

/// @brief A `qvector` is an owning, dynamically sized container for qudits.
/// The semantics of the `qvector` follows that of a `std::vector` for qudits.
/// It is templated on the number of levels for the held qudits.
template <std::size_t Levels = 2>
class qvector {
public:
  /// @brief Useful typedef indicating the underlying qudit type
  using value_type = qudit<Levels>;

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
  qvector(const state &state);
  qvector(const state *ptr);
  qvector(state *ptr);
  qvector(state &s);

  /// @brief `qvectors` cannot be copied
  qvector(qvector const &);

  /// @brief `qvectors` cannot be moved
  qvector(qvector &&);

  /// @brief `qvectors` cannot be copy assigned.
  qvector &operator=(const qvector &);

  /// @brief Iterator interface, begin.
  auto begin();

  /// @brief Iterator interface, end.
  auto end();

  /// @brief Returns the qudit at `idx`.
  value_type &operator[](const std::size_t idx);

  /// @brief Returns the `[0, count)` qudits as a non-owning `qview`.
  qview<Levels> front(std::size_t count);

  /// @brief Returns the first qudit.
  value_type &front();

  /// @brief Returns the `[count, size())` qudits as a non-owning `qview`
  qview<Levels> back(std::size_t count);

  /// @brief Returns the last qudit.
  value_type &back();

  /// @brief Returns the `[start, start+size)` qudits as a non-owning `qview`
  qview<Levels> slice(std::size_t start, std::size_t size);

  /// @brief Returns the number of contained qudits.
  std::size_t size() const;

  /// @brief Destroys all contained qudits. Postcondition: `size() == 0`.
  void clear();
};

#else
template <std::size_t Levels = 2>
class qvector {
private:
  /// @brief Reference to the held / owned vector of qudits.
  Array *qubitsArray;

public:
  /// @brief Construct a `qvector` with `size` qudits in the |0> state.
  qvector(std::size_t size)
      : qubitsArray(__quantum__rt__qubit_allocate_array(size)) {}

  /// @cond
  /// Nullary constructor
  /// meant to be used with `kernel_builder<cudaq::qvector<>>`
  qvector() { qubitsArray = nullptr; }
  /// @endcond

  /// @brief Construct a `qvector` from an input state vector.
  /// The number of qubits is determined by the size of the input vector.
  /// If `validate` is set, it will check the norm of input state vector.
  explicit qvector(const std::vector<complex> &vector, bool validate) {
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
    std::size_t numQubits = std::log2(vector.size());
    qubitsArray = __quantum__rt__qubit_allocate_array_with_state_complex64(
        numQubits, vector.data());
  }
  qvector(const std::vector<complex> &vector)
      : qvector(vector, /*validate=*/false){};

  qvector(const std::vector<double> &vector)
      : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}
  qvector(std::vector<double> &&vector)
      : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}
  qvector(const std::initializer_list<double> &list)
      : qvector(std::vector<complex>{list.begin(), list.end()}) {}
  qvector(const std::vector<float> &vector)
      : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}
  qvector(std::vector<float> &&vector)
      : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}
  qvector(const std::initializer_list<float> &list)
      : qvector(std::vector<complex>{list.begin(), list.end()}) {}
  qvector(const std::initializer_list<complex> &list)
      : qvector(std::vector<complex>{list.begin(), list.end()}) {}

  ~qvector() { __quantum__rt__qubit_release_array(qubitsArray); }

  /// @brief `qvectors` cannot be copied
  qvector(qvector const &) = delete;

  /// @brief `qvectors` cannot be moved
  qvector(qvector &&) = delete;

  /// @brief `qvectors` cannot be copy assigned.
  qvector &operator=(const qvector &) = delete;

  /// @brief Returns the qubit at `idx`.
  Qubit &operator[](const std::size_t idx) {
    return *reinterpret_cast<Qubit *>(
        __quantum__rt__array_get_element_ptr_1d(qubitsArray, idx));
  }

  /// @brief Returns the first qudit.
  Qubit &front() {
    return *reinterpret_cast<Qubit *>(
        __quantum__rt__array_get_element_ptr_1d(qubitsArray, 0));
  }

  /// @brief Destroys all contained qudits. Postcondition: `size() == 0`.
  void clear() {
    __quantum__rt__qubit_release_array(qubitsArray);
    qubitsArray = nullptr;
  }
};
#endif

} // namespace cudaq
