/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RuntimeBackendProvider.h"
#include "FmtCore.h"
#include "Logger.h"
#include "Registry.h"
#include "RuntimeTarget.h"
#include "cudaq/Support/TargetConfigYaml.h"
#include "cudaq/host_config.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/execution_manager.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/resourcecounter/ResourceCounter.h"
#include <dlfcn.h>
#include <fstream>
#include <llvm/ADT/StringSwitch.h>
#include <memory>
#include <mutex>
#include <ranges>
#include <regex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>

namespace {
// Access to the global provider singleton
std::recursive_mutex singletonMutex;
#define LOCK std::unique_lock<std::recursive_mutex> lock(singletonMutex)

#if defined(__APPLE__) && defined(__MACH__)
constexpr const char libSuffix[] = "dylib";
#else
constexpr const char libSuffix[] = "so";
#endif

using cudaq::RuntimeBackendProvider;
using LibraryHandle = RuntimeBackendProvider::LibraryHandle;
template <typename T>
using Factory = RuntimeBackendProvider::Factory<T>;
template <typename T>
using EntriesList = RuntimeBackendProvider::EntriesList<T>;

/// Join the keys of a map into a string, separated by commas.
template <typename Entries>
std::string joinEntryNames(const Entries &entries) {
  std::stringstream ss;
  bool first = true;
  for (const auto &entry : entries) {
    if (!first)
      ss << ", ";
    ss << entry.first;
    first = false;
  }
  return ss.str();
}

template <typename T>
Factory<T> wrapUniquePtrFactory(T *(*rawPtrFactory)()) {
  return [rawPtrFactory] { return std::unique_ptr<T>(rawPtrFactory()); };
}

/// @brief Populate map by looking for create<entryType> symbols in the
/// libraries.
/// @tparam T The entry type
/// @param entries The list of entries to populate
/// @param entryType The name of the entry type to populate
/// @param libHandles The map of library handles to populate
template <typename T>
void populateFromLibraries(
    EntriesList<Factory<T>> &entries, const std::string &entryType,
    const std::unordered_map<std::string, LibraryHandle> &libHandles) {
  std::string functionName = fmt::format("create{}", entryType);
  for (const auto &libPair : libHandles) {
    auto libHandle = libPair.second.get();
    void *symbol = dlsym(libHandle, functionName.c_str());
    if (symbol) {
      auto getNameFunctionName = fmt::format("get{}Name", entryType);
      auto getName = dlsym(libHandle, getNameFunctionName.c_str());
      std::string name;
      if (getName) {
        name = std::string(reinterpret_cast<char *(*)()>(getName)());
      } else {
        CUDAQ_WARN("Library {} does not define a name for {} instance.",
                   libPair.first, entryType);
        name = libPair.first;
      }
      auto rawPtrFactory = reinterpret_cast<T *(*)()>(symbol);
      entries.emplace_back(name, wrapUniquePtrFactory<T>(rawPtrFactory));
    }
  }
  if (!entries.empty()) {
    CUDAQ_INFO("Found {} {} entries: {}", entries.size(), entryType,
               joinEntryNames(entries));
  } else {
    CUDAQ_INFO("No {} entries found.", entryType);
  }
}

/// @brief Move an entry with the given name to the front of the list.
/// @tparam T The type stored in the EntriesList
/// @param entries The list of entries to modify
/// @param name The name of the entry to move to the front
/// @return true if the entry was found and moved, false if not found
template <typename T>
bool moveToFront(RuntimeBackendProvider::EntriesList<T> &entries,
                 const std::string &name) {
  auto it =
      std::find_if(entries.begin(), entries.end(),
                   [&](const auto &entry) { return entry.first == name; });
  if (it == entries.end()) {
    return false;
  } else if (it == entries.begin()) {
    return true;
  } else {
    std::rotate(entries.begin(), it, it + 1);
    return true;
  }
}

/// Get the simulator type for a given simulator name
RuntimeBackendProvider::SimulatorType
getSimulatorType(const std::string &name) {
  using SimulatorType = RuntimeBackendProvider::SimulatorType;
  return llvm::StringSwitch<SimulatorType>(name)
      .Case("resource_counter", SimulatorType::ResourceCounterSimulator)
      .Default(SimulatorType::CircuitSimulator);
}

/// Creates a LibraryHandle that automatically calls dlclose on destruction.
LibraryHandle createLibraryHandle(const char *path) {
  return LibraryHandle(dlopen(path, RTLD_GLOBAL | RTLD_NOW), [](void *h) {
    if (h)
      dlclose(h);
  });
}

static constexpr const char PLATFORM_LIBRARY[] = "PLATFORM_LIBRARY=";
static constexpr const char NVQIR_SIMULATION_BACKEND[] =
    "NVQIR_SIMULATION_BACKEND=";
static constexpr const char IS_FP64_SIMULATION[] =
    "CUDAQ_SIMULATION_SCALAR_FP64";

/// @brief Parse a runtime target config string and update the RuntimeTarget
void parseRuntimeTarget(const std::filesystem::path &cudaqLibPath,
                        cudaq::RuntimeTarget &target,
                        const std::string &nvqppBuildConfig) {
  cudaq::simulation_precision precision = cudaq::simulation_precision::fp32;
  std::optional<std::string> foundPlatformName, foundSimulatorName;
  for (auto &line : cudaq::split(nvqppBuildConfig, '\n')) {
    if (line.find(PLATFORM_LIBRARY) != std::string::npos) {
      cudaq::trim(line);
      auto platformName = cudaq::split(line, '=')[1];
      // Post-process the string
      platformName.erase(
          std::remove(platformName.begin(), platformName.end(), '\"'),
          platformName.end());
      platformName = std::regex_replace(platformName, std::regex("-"), "_");
      foundPlatformName = platformName;
    } else if (line.find(NVQIR_SIMULATION_BACKEND) != std::string::npos &&
               !foundSimulatorName.has_value()) {
      cudaq::trim(line);
      auto simulatorName = cudaq::split(line, '=')[1];
      // Post-process the string
      simulatorName.erase(
          std::remove(simulatorName.begin(), simulatorName.end(), '\"'),
          simulatorName.end());

      CUDAQ_DBG("CUDA-Q Library Path is {}.", cudaqLibPath.string());
      const auto libName =
          fmt::format("libnvqir-{}.{}", simulatorName, libSuffix);

      if (std::filesystem::exists(cudaqLibPath / libName)) {
        CUDAQ_DBG("Using {} simulator for target {}", simulatorName,
                  target.name);
        foundSimulatorName =
            std::regex_replace(simulatorName, std::regex("-"), "_");
      } else {
        CUDAQ_DBG(
            "Skipping {} simulator for target {} since it is not available",
            simulatorName, target.name);
      }
    } else if (line.find(IS_FP64_SIMULATION) != std::string::npos) {
      precision = cudaq::simulation_precision::fp64;
    }
  }
  target.platformName = foundPlatformName.value_or("default");
  target.simulatorName = foundSimulatorName.value_or("");
  target.precision = precision;
}

std::string
formatConfigForTarget(const std::map<std::string, std::string> &extraConfig,
                      const cudaq::RuntimeTarget &target) {
  std::string backendConfigStr = target.name;
  for (auto &[key, value] : extraConfig)
    backendConfigStr += fmt::format(";{};{}", key, value);
  return backendConfigStr;
}

/// @brief Load a RuntimeTarget from a YAML config file
cudaq::RuntimeTarget
loadTargetFromConfigFile(const std::filesystem::path &configPath) {
  // Open the file and parse YAML
  std::ifstream inFile(configPath.string());
  const std::string configFileContent((std::istreambuf_iterator<char>(inFile)),
                                      std::istreambuf_iterator<char>());
  cudaq::config::TargetConfig config;
  llvm::yaml::Input Input(configFileContent.c_str());
  Input >> config;

  // Extract target name from file path
  auto fileName = configPath.filename().string();
  const std::string configFileExt = ".yml";
  auto targetName = std::regex_replace(fileName, std::regex(configFileExt), "");

  CUDAQ_DBG("Found Target {} with config file {}", targetName, fileName);

  // Process runtime args to get config string
  const std::string defaultTargetConfigStr =
      cudaq::config::processRuntimeArgs(config, {});

  // Create RuntimeTarget
  cudaq::RuntimeTarget target;
  target.config = config;
  target.name = targetName;
  target.description = config.Description;

  // Parse the config string (need cudaqLibPath)
  auto cudaqLibPath = configPath.parent_path().parent_path() / "lib";
  parseRuntimeTarget(cudaqLibPath, target, defaultTargetConfigStr);

  return target;
}

std::string getSimulatorName(const cudaq::RuntimeTarget &target,
                             const cudaq::RuntimeTarget &defaultTarget) {
  std::string simName = target.simulatorName;
  if (simName.empty()) {
    // This target doesn't have a simulator defined, e.g., hardware targets.
    // We still need a simulator in case of local emulation.
    simName = defaultTarget.simulatorName;

    // This is really a user error
    if (simName.empty())
      throw std::runtime_error("Default target " + defaultTarget.name +
                               " doesn't define a simulator. Please check your "
                               "CUDAQ_DEFAULT_SIMULATOR environment variable.");
  }
  return simName;
}

/// Get the filename of a path without the extension.
std::string getFilenameWithoutExtension(const std::filesystem::path &path) {
  auto mutablePath = path;
  return mutablePath.replace_extension().filename().string();
}

} // namespace

namespace cudaq {

RuntimeBackendProvider &RuntimeBackendProvider::getSingleton() {
  LOCK;

  static RuntimeBackendProvider singleton;
  return singleton;
}

void RuntimeBackendProvider::initialize(
    const std::vector<std::filesystem::path> &dynamicLibPaths,
    const std::vector<std::filesystem::path> &targetConfigPaths) {
  CUDAQ_INFO("Initializing RuntimeBackendProvider");

  LOCK;

  if (libHandles_.size() > 0) {
    // Do not allow re-initialization, as this would lead to duplicate
    // entries in the registries.
    throw std::runtime_error("RuntimeBackendProvider already initialized.");
  }

  // Load all specified libraries
  if (loadLibrary(nullptr).empty()) {
    throw std::runtime_error("Failed to load statically linked libraries.");
  }
  for (const auto &path : dynamicLibPaths) {
    loadLibrary(&path);
  }
  CUDAQ_INFO("Loaded {} libraries dynamically", libHandles_.size() - 1);
  CUDAQ_DBG("Loaded libraries: {}", joinEntryNames(libHandles_));

  // Populate available components from the loaded libraries
  populateFromLibraries(availableSimulators_, "CircuitSimulator", libHandles_);
  populateFromLibraries(availablePlatforms_, "QuantumPlatform", libHandles_);
  populateFromLibraries(availableExecutionManagers_, "ExecutionManager",
                        libHandles_);

  // Parse target YAML configuration files
  for (const auto &configPath : targetConfigPaths) {
    auto target = loadTargetFromConfigFile(configPath);

    CUDAQ_DBG("Found Target: {} -> (sim={}, platform={})", target.name,
              target.simulatorName, target.platformName);

    // Add to available targets
    availableTargets_.emplace_back(target.name, target);
  }

  CUDAQ_INFO("RuntimeBackendProvider initialization complete. Registered {} "
             "targets, {} simulators, {} platforms, {} execution managers.",
             availableTargets_.size(), availableSimulators_.size(),
             availablePlatforms_.size(), availableExecutionManagers_.size());
}

// ===== quantum_platform management =====

namespace {
bool resetPlatform = false;
}

void RuntimeBackendProvider::setPlatform(const std::string &name) {
  LOCK;

  CUDAQ_DBG("Setting quantum platform");
  if (!moveToFront(availablePlatforms_, name)) {
    throw std::runtime_error("Failed to set platform: " + name + " not found");
  }
  CUDAQ_INFO("Set current platform to: {}", name);
  resetPlatform = true;
}

quantum_platform *RuntimeBackendProvider::getPlatform() const {
  static std::unique_ptr<quantum_platform> currentPlatform = nullptr;

  LOCK;

  if (resetPlatform) {
    currentPlatform.reset();
    resetPlatform = false;
  }
  if (!currentPlatform) {
    if (availablePlatforms_.empty()) {
      return nullptr;
    }
    currentPlatform =
        std::unique_ptr<quantum_platform>(availablePlatforms_.front().second());
  }
  // TODO: returning the raw pointer is prone to race conditions
  return currentPlatform.get();
}

// ===== CircuitSimulator management =====

namespace {
bool resetSimulator = false;
}

void RuntimeBackendProvider::setSimulator(const std::string &name) {
  LOCK;

  CUDAQ_DBG("Setting circuit simulator");
  if (!moveToFront(availableSimulators_, name)) {
    throw std::runtime_error("Failed to set simulator: " + name + " not found");
  }
  CUDAQ_INFO("Set current simulator to: {}", name);
  resetSimulator = true;
}

nvqir::CircuitSimulator *RuntimeBackendProvider::getSimulator() const {
  thread_local static std::unique_ptr<nvqir::CircuitSimulator>
      currentSimulator = nullptr;

  LOCK;

  if (resetSimulator) {
    currentSimulator.reset();
    resetSimulator = false;
  }
  if (!currentSimulator) {
    auto it = std::find_if(
        availableSimulators_.begin(), availableSimulators_.end(),
        [this](const auto &entry) {
          return getSimulatorType(entry.first) == currentSimulatorType_;
        });
    if (it == availableSimulators_.end()) {
      return nullptr;
    }
    currentSimulator = std::unique_ptr<nvqir::CircuitSimulator>(it->second());
  }
  return currentSimulator.get();
}

nvqir::ResourceCounter *
RuntimeBackendProvider::getResourceCounterSimulator() const {
  LOCK;

  if (currentSimulatorType_ != SimulatorType::ResourceCounterSimulator) {
    return nullptr;
  }
  return static_cast<nvqir::ResourceCounter *>(getSimulator());
}

void RuntimeBackendProvider::setSimulatorType(SimulatorType type) {
  LOCK;

  currentSimulatorType_ = type;
  resetSimulator = true;
}

RuntimeBackendProvider::SimulatorType
RuntimeBackendProvider::getCurrentSimulatorType() const {
  LOCK;

  return currentSimulatorType_;
}

// ===== ExecutionManager factory management =====

void RuntimeBackendProvider::setExecutionManager(const std::string &name) {
  LOCK;

  if (!moveToFront(availableExecutionManagers_, name)) {
    CUDAQ_WARN("Failed to set execution manager: " + name + " not found");
  } else {
    CUDAQ_INFO("Set current execution manager to: {}", name);
  }
}

std::unique_ptr<ExecutionManager>
RuntimeBackendProvider::createExecutionManager() {
  LOCK;

  if (availableExecutionManagers_.empty()) {
    CUDAQ_WARN("No execution manager found.");
    return nullptr;
  }
  return std::unique_ptr<ExecutionManager>(
      availableExecutionManagers_.front().second());
}

// ===== RuntimeTarget management =====

void RuntimeBackendProvider::setTarget(
    const std::string &name, std::map<std::string, std::string> extraConfig) {
  LOCK;

  auto it =
      std::find_if(availableTargets_.begin(), availableTargets_.end(),
                   [&](const auto &entry) { return entry.first == name; });
  if (it == availableTargets_.end())
    throw std::runtime_error("Invalid target name (" + name + ").");

  // Create a copy and parse the config string
  RuntimeTarget newTarget = it->second;

  if (!newTarget.config.WarningMsg.empty()) {
    fmt::print(fmt::fg(fmt::color::red), "[warning] ");
    fmt::print(fmt::fg(fmt::color::blue), "Target {}: {}\n", newTarget.name,
               newTarget.config.WarningMsg);
  }

  auto cudaqLibPath =
      std::filesystem::path(cudaq::getCUDAQLibraryPath()).parent_path();
  parseRuntimeTarget(cudaqLibPath, newTarget,
                     processRuntimeArgs(newTarget.config, extraConfig));

  CUDAQ_INFO("Setting target={} (sim={}, platform={})", newTarget.name,
             newTarget.simulatorName, newTarget.platformName);

  // Update the current target
  currentTarget_ = newTarget;
  loadMissingLibraries(cudaqLibPath);
  updateSimulator();
  updatePlatform(extraConfig);
  updateExecutionManager(cudaqLibPath);
}

void RuntimeBackendProvider::setDefaultTarget(const std::string &name) {
  LOCK;

  CUDAQ_INFO("Setting default target: {}", name);

  // Move target to front of list using moveToFront utility
  if (!moveToFront(availableTargets_, name)) {
    throw std::runtime_error("Failed to set default target: " + name +
                             " not found");
  }
}

RuntimeTarget RuntimeBackendProvider::getTarget() const {
  LOCK;

  return currentTarget_;
}

RuntimeTarget RuntimeBackendProvider::getTarget(const std::string &name) const {
  LOCK;

  auto it =
      std::find_if(availableTargets_.begin(), availableTargets_.end(),
                   [&](const auto &entry) { return entry.first == name; });
  if (it == availableTargets_.end())
    throw std::runtime_error("Invalid target name (" + name + ").");
  return it->second;
}

std::vector<RuntimeTarget> RuntimeBackendProvider::getTargets() const {
  LOCK;

  auto keys = std::views::keys(availableTargets_);
  return std::vector<RuntimeTarget>(keys.begin(), keys.end());
}

bool RuntimeBackendProvider::hasTarget(const std::string &name) const {
  LOCK;

  return std::find_if(availableTargets_.begin(), availableTargets_.end(),
                      [&](const auto &entry) { return entry.first == name; }) !=
         availableTargets_.end();
}

void RuntimeBackendProvider::resetTarget() {
  LOCK;

  if (availableTargets_.empty()) {
    return;
  }
  auto &name = availableTargets_.front().first;

  CUDAQ_INFO("Resetting to default target: {}", name);
  setTarget(name, {});
}

bool RuntimeBackendProvider::hasSimulator(const std::string &name) const {
  LOCK;

  return std::find_if(availableSimulators_.begin(), availableSimulators_.end(),
                      [&](const auto &entry) { return entry.first == name; }) !=
         availableSimulators_.end();
}

bool RuntimeBackendProvider::hasPlatform(const std::string &name) const {
  LOCK;

  return std::find_if(availablePlatforms_.begin(), availablePlatforms_.end(),
                      [&](const auto &entry) { return entry.first == name; }) !=
         availablePlatforms_.end();
}

std::string
RuntimeBackendProvider::loadLibrary(const std::filesystem::path *path) {
  LOCK;

  const char *pathString = path ? path->c_str() : nullptr;
  LibraryHandle libHandle = createLibraryHandle(pathString);

  if (!libHandle) {
    char *error_msg = dlerror();
    CUDAQ_INFO("Failed to load '{}': ERROR '{}'",
               pathString ? pathString : "nullptr",
               error_msg ? std::string(error_msg) : "unknown.");
    return "";
  }

  CUDAQ_DBG("Successfully loaded library: {}",
            pathString ? pathString : "nullptr");
  auto libName = path ? getFilenameWithoutExtension(*path)
                      : std::string("static_loaded_libraries");
  libHandles_.emplace(libName, std::move(libHandle));
  return libName;
}

void RuntimeBackendProvider::unloadLibrary(const std::string &name) {
  LOCK;

  libHandles_.erase(name);
}

void RuntimeBackendProvider::updateSimulator() {
  LOCK;

  auto simName =
      getSimulatorName(currentTarget_, availableTargets_.front().second);
  setSimulator(simName);
}

void RuntimeBackendProvider::updatePlatform(
    const std::map<std::string, std::string> &extraConfig) {
  LOCK;

  auto platformName = currentTarget_.platformName;
  setPlatform(platformName);
  getPlatform()->setTargetBackend(
      formatConfigForTarget(extraConfig, currentTarget_));
}

void RuntimeBackendProvider::updateExecutionManager(
    const std::filesystem::path &cudaqLibPath) {
  LOCK;

  constexpr const char *photonicsLibName = "libcudaq-em-photonics";
  std::string newExecManagerName;
  if ("orca-photonics" == currentTarget_.name) {
    auto libPath =
        cudaqLibPath / fmt::format("{}.{}", photonicsLibName, libSuffix);
    loadLibrary(&libPath);
    newExecManagerName = "photonics";
  } else {
    unloadLibrary(photonicsLibName);
    newExecManagerName = "default";
  }

  // repopulate the list of execution manager
  availableExecutionManagers_.clear();
  populateFromLibraries(availableExecutionManagers_, "ExecutionManager",
                        libHandles_);
  setExecutionManager(newExecManagerName);
}

void RuntimeBackendProvider::loadMissingLibraries(
    const std::filesystem::path &cudaqLibPath) {
  LOCK;

  auto potentialServerHelperPath =
      cudaqLibPath / fmt::format("libcudaq-serverhelper-{}.{}",
                                 currentTarget_.name, libSuffix);
  if (std::filesystem::exists(potentialServerHelperPath) &&
      libHandles_.find(potentialServerHelperPath.string()) ==
          libHandles_.end()) {
    auto serverHelperHandle =
        createLibraryHandle(potentialServerHelperPath.string().c_str());
    if (serverHelperHandle)
      libHandles_.emplace(potentialServerHelperPath.string(),
                          std::move(serverHelperHandle));
  }
}

} // namespace cudaq
