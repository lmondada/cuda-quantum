/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "RuntimeTarget.h"
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
namespace nvqir {
class CircuitSimulator;
class ResourceCounter;
} // namespace nvqir

namespace cudaq {

// Forward declarations
class quantum_platform;
class ExecutionManager;

namespace python {
class PythonBackendProvider;
} // namespace python

namespace __internal__ {
class TargetSetter;
} // namespace __internal__

/// @brief The RuntimeBackendProvider provides a centralized dependency
/// injection container for managing quantum backends and runtime components.
/// It supports both dynamic loading (Python) and static linking (C++) modes.
class RuntimeBackendProvider {
public:
  /// @brief The type of simulator currently in use
  enum class SimulatorType {
    CircuitSimulator,
    ResourceCounterSimulator,
  };
  /// Unique pointer type for managing library handles with automatic unloading.
  using LibraryHandle = std::unique_ptr<void, std::function<void(void *)>>;
  template <typename T>
  using EntriesList = std::vector<std::pair<std::string, T>>;
  template <typename T>
  using Factory = std::function<std::unique_ptr<T>()>;

  /// Get the singleton instance.
  ///
  /// This will lock a mutex for the lifetime of the returned handle, so be
  /// mindful of its lifetime.
  static RuntimeBackendProvider &getSingleton();

  // Delete copy/move constructors and assignment operators
  RuntimeBackendProvider(const RuntimeBackendProvider &) = delete;
  RuntimeBackendProvider &operator=(const RuntimeBackendProvider &) = delete;
  RuntimeBackendProvider(RuntimeBackendProvider &&) = delete;
  RuntimeBackendProvider &operator=(RuntimeBackendProvider &&) = delete;

  /// @brief Initialize the provider.
  ///
  /// Loads simulators and platforms from symbols linked at compile time as well
  /// as the dynamic libraries specified. Optionally loads targets from YAML
  /// files.
  /// @param dynamicLibPaths Paths to dynamic libraries to load
  /// @param targetConfigPaths Paths to target YAML configuration files
  /// @param defaultTarget Optional name of the default target (if nullopt, uses
  /// first)
  void
  initialize(const std::vector<std::filesystem::path> &dynamicLibPaths = {},
             const std::vector<std::filesystem::path> &targetConfigPaths = {});

  // ===== quantum_platform management =====

  /// @brief Set the current quantum platform instance by name
  void setPlatform(const std::string &name);

  /// @brief Get the current quantum platform instance
  /// @return Pointer to the quantum platform, or nullptr if not set
  quantum_platform *getPlatform() const;

  // ===== CircuitSimulator management =====

  /// @brief Set the current circuit simulator instance
  void setSimulator(const std::string &name);

  /// @brief Get the current circuit simulator instance
  /// @return Pointer to the circuit simulator, or nullptr if not set
  nvqir::CircuitSimulator *getSimulator() const;

  /// @brief Get the current simulator instance cast as ResourceCounter, if it
  /// is a ResourceCounterSimulator.
  nvqir::ResourceCounter *getResourceCounterSimulator() const;

  /// @brief Set the current simulator type
  ///
  /// Currently supporting CircuitSimulator and ResourceCounter types.
  void setSimulatorType(SimulatorType type = SimulatorType::CircuitSimulator);

  /// @brief Get the current simulator type
  SimulatorType getCurrentSimulatorType() const;

  // ===== ExecutionManager factory management =====

  /// @brief Set the factory function for creating ExecutionManager instances
  void setExecutionManager(const std::string &name = "default");

  /// @brief Create a new ExecutionManager instance using the registered factory
  /// @return Unique pointer to a new ExecutionManager, or nullptr if no factory
  std::unique_ptr<ExecutionManager> createExecutionManager();

  // ===== RuntimeTarget registry =====

  /// @brief Get the current target
  /// @return Current RuntimeTarget
  RuntimeTarget getTarget() const;

  /// @brief Get a target by name
  /// @param name Target name
  /// @return RuntimeTarget with the given name
  /// @throws std::runtime_error if target not found
  RuntimeTarget getTarget(const std::string &name) const;

  /// @brief Get all available targets
  /// @return Vector of all RuntimeTargets
  std::vector<RuntimeTarget> getTargets() const;

  /// @brief Check if a target with the given name exists
  /// @param name Target name
  /// @return true if target exists
  bool hasTarget(const std::string &name) const;

  /// @brief Check if a simulator with the given name is available
  bool hasSimulator(const std::string &name) const;

  /// @brief Check if a platform with the given name is available
  bool hasPlatform(const std::string &name) const;

  /// @brief Load a library into the provider.
  ///
  /// If nullptr, return a handle to all loaded libraries.
  ///
  /// Returns the filename of the library, or an empty string if the library
  /// could not be loaded.
  std::string loadLibrary(const std::filesystem::path *path = nullptr);

  /// @brief Unload a library from the provider
  void unloadLibrary(const std::string &name);

private:
  friend class python::PythonBackendProvider;
  friend class cudaq::__internal__::TargetSetter;

  // ===== Methods to be accessed by friend classes =====

  /// @brief Set the current target by name with config string
  /// @param name Target name
  /// @param targetConfigStr Target configuration string
  void setTarget(const std::string &name,
                 std::map<std::string, std::string> extraConfig);

  /// @brief Reset current target to default (first in list)
  void resetTarget();

  /// @brief Override the default target (moves to front of availableTargets_)
  /// @param name Target name
  void setDefaultTarget(const std::string &name);

  // ===== Internal helper methods =====

  RuntimeBackendProvider() = default;
  ~RuntimeBackendProvider() = default;

  /// Update the simulator, platform, and execution manager factory based on the
  /// current target
  void updateSimulator();
  void updatePlatform(const std::map<std::string, std::string> &extraConfig);
  void updateExecutionManager(const std::filesystem::path &cudaqLibPath);
  void loadMissingLibraries(const std::filesystem::path &cudaqLibPath);

  // Current simulator type
  SimulatorType currentSimulatorType_ = SimulatorType::CircuitSimulator;

  // Current target
  RuntimeTarget currentTarget_;

  // Available loaded simulators, platforms, execution managers and targets
  // - Simulators, platforms, and execution managers are ordered by the most
  //   recently used.
  // - Targets store the default target in the front of the list.
  EntriesList<Factory<nvqir::CircuitSimulator>> availableSimulators_;
  EntriesList<Factory<quantum_platform>> availablePlatforms_;
  EntriesList<Factory<ExecutionManager>> availableExecutionManagers_;
  EntriesList<RuntimeTarget> availableTargets_;

  // Dynamically loaded libraries, keyed by path, to be closed on destruction
  std::unordered_map<std::string, LibraryHandle> libHandles_;
};

} // namespace cudaq
