/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DefaultQPU.h"
#include "common/ExecutionContext.h"
#include "common/Timing.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/runtime/logger/logger.h"

/// This file defines the default, library mode, quantum platform. Its goal is
/// to create a single QPU that is added to the quantum_platform which delegates
/// kernel execution to the current Execution Manager.

using namespace cudaq;

namespace {
/// The DefaultQuantumPlatform is a quantum_platform that provides a single
/// simulated QPU, which delegates to the QIS ExecutionManager.
class DefaultQuantumPlatform : public cudaq::quantum_platform {
public:
  DefaultQuantumPlatform() {
    // Populate the information and add the QPUs
    addQPU(std::make_unique<cudaq::DefaultQPU>());
  }

private:
  /// @brief Set the target backend. Here we have an opportunity to know the
  /// -qpu QPU target we are running on. This function will read in the qpu
  /// configuration file and search for the PLATFORM_QPU variable, and if found,
  /// will change from the DefaultQPU to the QPU subtype specified by that
  /// variable.
  void setTargetBackend(const std::string &backend) override {

    CUDAQ_INFO("Backend string is {}", backend);
    auto [targetName, configMap] =
        cudaq::detail::parseBackendConfigString(backend);
    auto config = cudaq::detail::loadBackendTargetConfig(backend);

    std::unique_ptr<cudaq::QPU> newQPU;
    const bool usesPlatformQpu = config.BackendConfig.has_value() &&
                                 !config.BackendConfig->PlatformQpu.empty();
    if (usesPlatformQpu) {
      auto qpuName = config.BackendConfig->PlatformQpu;
      CUDAQ_INFO("Default platform QPU subtype name: {}", qpuName);
      newQPU = cudaq::registry::get<cudaq::QPU>(qpuName);
      if (newQPU == nullptr)
        throw std::runtime_error(
            qpuName + " is not a valid QPU name for the default platform.");
    } else {
      newQPU = std::make_unique<cudaq::DefaultQPU>();
    }

    // Forward to the QPU so it can materialize its own compile target.
    newQPU->setTargetBackend(backend);

    CompileTarget compileTarget;
    if (usesPlatformQpu) {
      compileTarget = newQPU->getCompileTarget();
    } else {
      compileTarget = createDefaultCompileTarget(config, configMap);
      compileTarget.fullySpecialize = false;
    }

    auto endpoint = RuntimeEndpoint::fromQPU(std::move(newQPU));
    cudaq::detail::applyTargetMetadata(endpoint, config, targetName);
    clearQPUs();
    addQPU(compileTarget, endpoint);
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(DefaultQuantumPlatform, default)
