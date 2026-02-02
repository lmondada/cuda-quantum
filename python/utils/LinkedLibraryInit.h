/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Resources.h"
#include "common/RuntimeTarget.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace nvqir {
void setChoiceFunction(std::function<bool()> choice);
cudaq::Resources *getResourceCounts();
} // namespace nvqir

namespace cudaq::python {

/// @brief Initialize the RuntimeBackendProvider for Python usage
/// Discovers targets, simulators, platforms, and configures defaults
void initializeBackendProvider();

/// @brief Set the current target with optional configuration
/// @param name Target name
/// @param config Additional key-value configuration pairs
void setTarget(const std::string &name,
               std::map<std::string, std::string> config = {});

/// @brief Get the current target
RuntimeTarget getTarget();

/// @brief Get a target by name
/// @param name Target name
RuntimeTarget getTarget(const std::string &name);

/// @brief Get all available targets
std::vector<RuntimeTarget> getTargets();

/// @brief Check if a target exists
/// @param name Target name
bool hasTarget(const std::string &name);

/// @brief Reset to the default target
void resetTarget();

/// @brief Get the transport layer for the current target
std::string getTransportLayer();

namespace detail {
void switchToResourceCounterSimulator();
void stopUsingResourceCounterSimulator();
void setChoiceFunction(std::function<bool()> choice);
cudaq::Resources *getResourceCounts();
} // namespace detail

} // namespace cudaq::python
