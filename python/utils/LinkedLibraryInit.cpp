/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LinkedLibraryInit.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/RuntimeBackendProvider.h"
#include "cudaq/target_control.h"
#include "cudaq/utils/cudaq_utils.h"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

using SimulatorType = cudaq::RuntimeBackendProvider::SimulatorType;

namespace {

/// @brief A utility function to check availability of Nvidia GPUs and return
/// their count.
int countGPUs() {
  int retCode = std::system("nvidia-smi >/dev/null 2>&1");
  if (0 != retCode) {
    CUDAQ_INFO("nvidia-smi: command not found");
    return -1;
  }

  char tmpFile[] = "/tmp/.cmd.capture.XXXXXX";
  int fileDescriptor = mkstemp(tmpFile);
  if (-1 == fileDescriptor) {
    CUDAQ_INFO("Failed to create a temporary file to capture output");
    return -1;
  }

  std::string command = "nvidia-smi -L 2>/dev/null | wc -l >> ";
  command.append(tmpFile);
  retCode = std::system(command.c_str());
  if (0 != retCode) {
    CUDAQ_INFO("Encountered error while invoking 'nvidia-smi'");
    return -1;
  }

  std::stringstream buffer;
  buffer << std::ifstream(tmpFile).rdbuf();
  close(fileDescriptor);
  unlink(tmpFile);
  return std::stoi(buffer.str());
}

/// @brief Discover target configuration files in the targets directory
std::vector<std::filesystem::path>
discoverTargetConfigPaths(const std::filesystem::path &targetPath) {
  std::vector<std::filesystem::path> configPaths;

  // directory_iterator ordering is unspecified, so sort it to make it
  // repeatable and consistent.
  std::vector<std::filesystem::directory_entry> targetEntries;
  for (const auto &entry : std::filesystem::directory_iterator{targetPath})
    targetEntries.push_back(entry);
  std::sort(targetEntries.begin(), targetEntries.end(),
            [](const std::filesystem::directory_entry &a,
               const std::filesystem::directory_entry &b) {
              return a.path().filename() < b.path().filename();
            });

  // Loop over all target files
  for (const auto &configFile : targetEntries) {
    auto path = configFile.path();
    // They must have a .yml suffix
    const std::string configFileExt = ".yml";
    if (path.extension().string() == configFileExt) {
      configPaths.push_back(path);
    }
  }

  return configPaths;
}

bool filenameStartsWith(const std::filesystem::directory_entry &entry,
                        const std::string &prefix) {
  return entry.path().filename().string().rfind(prefix, 0) == 0;
}

/// @brief Determine the default target based on GPU availability and
/// environment variables.
std::string determineDefaultTarget(cudaq::RuntimeBackendProvider &provider) {
  std::string defaultTarget = "qpp-cpu";

  if (countGPUs() > 0) {
    // Check if nvidia target is available and has required simulator
    defaultTarget = "nvidia";
    if (provider.hasTarget("nvidia")) {
      auto target = provider.getTarget("nvidia");
      if (!provider.hasSimulator(target.simulatorName)) {
        CUDAQ_INFO(
            "GPU(s) found but cannot select nvidia target because simulator "
            "is not available. Are all dependencies installed?");
        defaultTarget = "qpp-cpu";
      }
    } else {
      CUDAQ_INFO("GPU(s) found but cannot select nvidia target because nvidia "
                 "target not found.");
      // Reset to qpp-cpu if nvidia doesn't exist
      if (provider.hasTarget("qpp-cpu")) {
        defaultTarget = "qpp-cpu";
      }
    }
  }

  // Handle CUDAQ_DEFAULT_SIMULATOR environment variable
  auto env = std::getenv("CUDAQ_DEFAULT_SIMULATOR");
  if (env) {
    CUDAQ_INFO("'CUDAQ_DEFAULT_SIMULATOR' = {}", env);
    if (provider.hasTarget(env)) {
      CUDAQ_INFO("Valid target");
      defaultTarget = env;
    }
  }

  return defaultTarget;
}

} // anonymous namespace

namespace cudaq::python {

void initializeBackendProvider() {
  CUDAQ_INFO("Init infrastructure for pythonic builder.");

  if (!cudaq::__internal__::canModifyTarget())
    return;

  cudaq::__internal__::CUDAQLibraryData data;
#if defined(__APPLE__) && defined(__MACH__)
  auto libSuffix = "dylib";
  cudaq::__internal__::getCUDAQLibraryPath(&data);
#else
  auto libSuffix = "so";
  dl_iterate_phdr(cudaq::__internal__::getCUDAQLibraryPath, &data);
#endif

  std::filesystem::path nvqirLibPath{data.path};
  auto cudaqLibPath = nvqirLibPath.parent_path();
  if (cudaqLibPath.filename().string() == "common") {
    // this is a build path
    cudaqLibPath = cudaqLibPath.parent_path().parent_path() / "lib";
  }

  // Discover target configuration files
  auto targetPath = cudaqLibPath.parent_path() / "targets";
  auto targetConfigPaths = discoverTargetConfigPaths(targetPath);

  CUDAQ_INFO("Init: Library Path is {}.", cudaqLibPath.string());

  // We have to ensure that nvqir and cudaq are loaded
  std::vector<std::filesystem::path> libPaths{
      cudaqLibPath / fmt::format("libnvqir.{}", libSuffix),
      cudaqLibPath / fmt::format("libcudaq.{}", libSuffix)};

  const char *dynlibs_var = std::getenv("CUDAQ_DYNLIBS");
  if (dynlibs_var != nullptr) {
    std::string dynlib;
    std::stringstream ss((std::string(dynlibs_var)));
    while (std::getline(ss, dynlib, ':')) {
      CUDAQ_INFO("Init: add dynamic library path {}.", dynlib);
      libPaths.push_back(dynlib);
    }
  }

  // Search for all library files that are simulators or platforms
  std::vector<std::filesystem::directory_entry> entries;
  for (const auto &entry : std::filesystem::directory_iterator{cudaqLibPath}) {
    if (filenameStartsWith(entry, "libnvqir-") ||
        filenameStartsWith(entry, "libcudaq-platform-") ||
        filenameStartsWith(entry, "libcudaq-em-")) {
      entries.push_back(entry);
    }
  }
  // directory_iterator ordering is unspecified, so sort it to make it
  // repeatable and consistent.
  std::sort(entries.begin(), entries.end(),
            [](const std::filesystem::directory_entry &a,
               const std::filesystem::directory_entry &b) {
              return a.path().filename() < b.path().filename();
            });
  libPaths.insert(libPaths.end(), entries.begin(), entries.end());

  // Initialize the RuntimeBackendProvider
  auto &provider = RuntimeBackendProvider::getSingleton();
  provider.initialize(libPaths, targetConfigPaths);

  // Determine the default target based on GPU availability and env vars
  std::string defaultTarget = determineDefaultTarget(provider);

  // Reset to default target
  provider.setDefaultTarget(defaultTarget);
  provider.resetTarget();
}

void setTarget(const std::string &targetName,
               std::map<std::string, std::string> extraConfig) {
  // Do not set the target if the disallow flag has been set.
  if (!cudaq::__internal__::canModifyTarget())
    return;

  auto &provider = RuntimeBackendProvider::getSingleton();

  if (!provider.hasTarget(targetName))
    throw std::runtime_error("Invalid target name (" + targetName + ").");

  // Set the target in the provider (this parses the config string)
  provider.setTarget(targetName, extraConfig);
}

RuntimeTarget getTarget() {
  auto &provider = RuntimeBackendProvider::getSingleton();
  return provider.getTarget();
}

RuntimeTarget getTarget(const std::string &name) {
  auto &provider = RuntimeBackendProvider::getSingleton();
  return provider.getTarget(name);
}

std::vector<RuntimeTarget> getTargets() {
  auto &provider = RuntimeBackendProvider::getSingleton();
  return provider.getTargets();
}

bool hasTarget(const std::string &name) {
  auto &provider = RuntimeBackendProvider::getSingleton();
  return provider.hasTarget(name);
}

void resetTarget() {
  auto &provider = RuntimeBackendProvider::getSingleton();
  provider.resetTarget();
}

std::string getTransportLayer() {
  if (cudaq::__internal__::canModifyTarget()) {
    auto target = getTarget();
    const std::string codegenEmission =
        target.config.getCodeGenSpec(target.runtimeConfig);
    if (!codegenEmission.empty())
      return codegenEmission;
  }
  // Default is full QIR.
  return "qir:0.1";
}

namespace detail {
void switchToResourceCounterSimulator() {
  RuntimeBackendProvider::getSingleton().setSimulatorType(
      SimulatorType::ResourceCounterSimulator);
}

void stopUsingResourceCounterSimulator() {
  RuntimeBackendProvider::getSingleton().setSimulatorType(
      SimulatorType::CircuitSimulator);
}

void setChoiceFunction(std::function<bool()> choice) {
  nvqir::setChoiceFunction(choice);
}

Resources *getResourceCounts() { return nvqir::getResourceCounts(); }
} // namespace detail

} // namespace cudaq::python
