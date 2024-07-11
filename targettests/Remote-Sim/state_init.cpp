/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu %s -o %t && %t
// RUN: nvq++ %cpp_std --target remote-mqpu %s -o %t && %t // TODO: this fails to compile, do we need it?
// clang-format on

#include <cudaq.h>
#include <iostream>


__qpu__ void test_complex_array_param(cudaq::state* inState) {
  cudaq::qvector q1(inState);
}

void printCounts(cudaq::sample_result& result) {
  std::vector<std::string> values{};
  for (auto &&[bits, counts] : result) {
    values.push_back(bits);
  }

  std::sort(values.begin(), values.end());
  for (auto &&bits : values) {
    std::cout << bits << '\n';
  }
}

int main() {
    {
      std::vector<cudaq::complex> vec{M_SQRT1_2, M_SQRT1_2, 0., 0.};
      std::vector<cudaq::complex> vec1{0., 0., M_SQRT1_2, M_SQRT1_2};
      auto state = cudaq::state::from_data(vec);
      auto state1 = cudaq::state::from_data(vec1);
      {
          // Passing state data as argument (kernel mode)
          auto counts = cudaq::sample(test_complex_array_param, &state);
          printCounts(counts);

          counts = cudaq::sample(test_complex_array_param, &state1);
          printCounts(counts);
      }

      // {
      //     // Passing state data as argument (builder mode)
      //     auto [kernel, state] = cudaq::make_kernel<cudaq::state*>();
      //     auto qubits = kernel.qalloc(state);

      //     auto counts = cudaq::sample(kernel, &state);
      //     printCounts(counts);

      //     counts = cudaq::sample(kernel, &state1);
      //     printCounts(counts);
      // }
    }
}

// CHECK: 00
// CHECK: 10

// CHECK: 0001
// CHECK: 0011
// CHECK: 1001
// CHECK: 1011

// CHECK: 00
// CHECK: 10

// CHECK: 00
// CHECK: 10

// CHECK: 00
// CHECK: 10

// CHECK: 01
// CHECK: 11

// CHECK: 00
// CHECK: 10

// CHECK: 01
// CHECK: 11

// CHECK: 00
// CHECK: 10

// CHECK: 01
// CHECK: 11
