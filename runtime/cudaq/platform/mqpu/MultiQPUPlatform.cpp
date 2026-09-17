/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DefaultQPU.h"
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "helpers/MQPUUtils.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/simulators.h"
#include <filesystem>
#include <map>

// Note: LLVM_INSTANTIATE_REGISTRY(cudaq::QPU::RegistryType) is intentionally
// NOT placed here. The canonical QPU registry instance lives in
// quantum_platform.cpp (libcudaq). With LLVM 22's static-inline Head/Tail
// pointers in llvm::Registry, having the instantiation in multiple DSOs can
// cause registry fragmentation — nodes added via cudaq_add_qpu_node (which
// targets libcudaq's registry) would be invisible to code in this DSO if the
// linker kept separate copies. A single instantiation in libcudaq avoids this.

namespace {
class MultiQPUQuantumPlatform : public cudaq::quantum_platform {

public:
  ~MultiQPUQuantumPlatform() {
    // Make sure that we clean up the client QPUs first before cleaning up the
    // remote servers.
    clearQPUs();
  }

  MultiQPUQuantumPlatform() { populateDefaultQPUs(); }

  bool supports_task_distribution() const override { return true; }

  void beginExecution() override {
    // Only set the CUDA device when GPU-backed QPUs are active.
    // Non-GPU platforms (e.g. ORCA) that replace the default QPUs
    // via setTargetBackend do not require a CUDA device assignment.
    auto qid = cudaq::getCurrentQpuId();
    int nDevices = cudaq::getCudaDeviceCount();
    if (nDevices > 0)
      cudaq::setCudaDevice(qid);
    // Base implementation of beginExecution will be called after this.
    cudaq::quantum_platform::beginExecution();
  }

private:
  void populateDefaultQPUs(
      const cudaq::config::TargetConfig &config = {},
      const std::map<std::string, std::string> &runtimeConfig = {},
      const std::string &targetName = {});

  static std::string getOption(const std::string &str,
                               const std::string &prefix) {
    // Return the first key-value configuration option found in the format:
    // "<prefix>;<option>".
    // Note: This expects an exact match of the prefix and the option value is
    // the next one.
    return cudaq::detail::getBackendConfigOption(str, prefix).value_or("");
  }

  static std::string formatUrl(const std::string &url) {
    auto formatted = url;
    // Default to http:// if none provided.
    if (!formatted.starts_with("http"))
      formatted = std::string("http://") + formatted;
    if (!formatted.empty() && formatted.back() != '/')
      formatted += '/';
    return formatted;
  }

  void setTargetBackend(const std::string &description) override {
    auto [targetName, runtimeConfig] =
        cudaq::detail::parseBackendConfigString(description);
    auto config = cudaq::detail::loadBackendTargetConfig(description);
    const std::string qpuSubType =
        config.BackendConfig.has_value() ? config.BackendConfig->PlatformQpu
                                         : std::string{};
    if (!qpuSubType.empty()) {
      if (!cudaq::registry::isRegistered<cudaq::QPU>(qpuSubType))
        throw std::runtime_error(
            fmt::format("Unable to retrieve {} QPU implementation. Please "
                        "check your installation.",
                        qpuSubType));
      if (qpuSubType == "orca") {
        auto urls = cudaq::split(getOption(description, "url"), ',');
        clearQPUs();
        for (std::size_t qId = 0; qId < urls.size(); ++qId) {
          auto newQPU = cudaq::registry::get<cudaq::QPU>("orca");
          newQPU->setId(qId);
          const std::string configStr =
              fmt::format("orca;url;{}", formatUrl(urls[qId]));
          newQPU->setTargetBackend(configStr);
          auto compileTarget = newQPU->getCompileTarget();
          auto endpoint = cudaq::RuntimeEndpoint::fromQPU(std::move(newQPU));
          cudaq::detail::applyTargetMetadata(endpoint, config, targetName);
          addQPU(compileTarget, endpoint);
        }
        return;
      } else {
        throw std::runtime_error(
            fmt::format("Unsupported platform QPU sub-type '{}' specified in "
                        "target config. Currently only 'orca' is supported.",
                        qpuSubType));
      }
    } else {
      populateDefaultQPUs(config, runtimeConfig, targetName);

      if (num_qpus() == 0) {
        // No QPU (GPU simulator nor specified platform QPU) was able to be
        // initialized, so we can't run.
        throw std::runtime_error(
            "No platform QPU implementations available. Please check your "
            "installation and target configuration.");
      }
    }
  }
};

void MultiQPUQuantumPlatform::populateDefaultQPUs(
    const cudaq::config::TargetConfig &config,
    const std::map<std::string, std::string> &runtimeConfig,
    const std::string &targetName) {
  clearQPUs();
  int nDevices = cudaq::getCudaDeviceCount();
  // Skipped if CUDA-Q was built with CUDA but no devices present at
  // runtime.
  if (nDevices > 0) {
    const char *envVal = std::getenv("CUDAQ_MQPU_NGPUS");
    if (envVal != nullptr) {
      int specifiedNDevices = 0;
      try {
        specifiedNDevices = std::stoi(envVal);
      } catch (...) {
        throw std::runtime_error("Invalid CUDAQ_MQPU_NGPUS environment "
                                 "variable, must be integer.");
      }

      if (specifiedNDevices < nDevices)
        nDevices = specifiedNDevices;
    }

    if (nDevices == 0)
      throw std::runtime_error("No GPUs available to instantiate platform.");

    auto compileTarget = cudaq::createDefaultCompileTarget(config, runtimeConfig);
    compileTarget.fullySpecialize = false;
    // Add a QPU for each GPU.
    for (int i = 0; i < nDevices; i++) {
      auto qpu = std::make_unique<cudaq::DefaultQPU>();
      qpu->setId(i);
      auto endpoint = cudaq::RuntimeEndpoint::fromQPU(std::move(qpu));
      cudaq::detail::applyTargetMetadata(endpoint, config, targetName);
      addQPU(compileTarget, endpoint);
    }
  }
}
} // namespace

CUDAQ_REGISTER_PLATFORM(MultiQPUQuantumPlatform, mqpu)
