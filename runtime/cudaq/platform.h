/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/RuntimeBackendProvider.h"
#include "cudaq/platform/quantum_platform.h"

namespace cudaq {

/// @brief Return the quantum platform provided by the linked platform library
/// @return
inline quantum_platform &get_platform() {
  auto &provider = RuntimeBackendProvider::getSingleton();
  return *provider.getPlatform();
}

/// @brief Return the number of QPUs (at runtime)
inline std::size_t platform_num_qpus() {
  auto &provider = RuntimeBackendProvider::getSingleton();
  return provider.getPlatform()->num_qpus();
}

/// @brief Return true if the quantum platform is remote.
inline bool is_remote_platform() {
  auto &provider = RuntimeBackendProvider::getSingleton();
  return provider.getPlatform()->is_remote();
}

/// @brief Return true if the quantum platform is a remote simulator.
inline bool is_remote_simulator_platform() {
  auto &provider = RuntimeBackendProvider::getSingleton();
  return provider.getPlatform()->get_remote_capabilities().isRemoteSimulator;
}

/// @brief Return true if the quantum platform is emulated.
inline bool is_emulated_platform() {
  auto &provider = RuntimeBackendProvider::getSingleton();
  return provider.getPlatform()->is_emulated();
}

// Declare this function, implemented elsewhere
std::string getQIR(const std::string &);

} // namespace cudaq
