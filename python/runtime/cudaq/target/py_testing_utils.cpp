/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "py_testing_utils.h"
#include "common/RuntimeBackendProvider.h"
#include "cudaq.h"
#include "cudaq/platform.h"
#include "nvqir/CircuitSimulator.h"
#include <pybind11/stl.h>

namespace nvqir {
void toggleDynamicQubitManagement();
} // namespace nvqir

namespace cudaq {

void bindTestUtils(py::module &mod) {
  auto testingSubmodule = mod.def_submodule("testing");

  testingSubmodule.def(
      "toggleDynamicQubitManagement",
      []() { nvqir::toggleDynamicQubitManagement(); }, "");

  testingSubmodule.def(
      "allocateQubits",
      [](std::size_t numQubits) {
        auto &provider = cudaq::RuntimeBackendProvider::getSingleton();
        auto *simulator = provider.getSimulator();
        if (!simulator)
          throw std::runtime_error(
              "No simulator available. Is backend initialized?");
        return simulator->allocateQubits(numQubits);
      },
      py::arg("numQubits"));

  testingSubmodule.def(
      "deallocateQubits", [](const std::vector<std::size_t> &qubits) {
        auto &provider = cudaq::RuntimeBackendProvider::getSingleton();
        auto *simulator = provider.getSimulator();
        if (!simulator)
          throw std::runtime_error(
              "No simulator available. Is backend initialized?");
        simulator->deallocateQubits(qubits);
      });

  testingSubmodule.def("getAndClearOutputLog", []() {
    auto &provider = cudaq::RuntimeBackendProvider::getSingleton();
    auto *simulator = provider.getSimulator();
    if (!simulator)
      throw std::runtime_error(
          "No simulator available. Is backend initialized?");
    auto log = simulator->outputLog;
    simulator->outputLog.clear();
    return log;
  });
}

} // namespace cudaq
