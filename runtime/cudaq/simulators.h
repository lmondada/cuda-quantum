/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/RuntimeBackendProvider.h"
#include "nvqir/CircuitSimulator.h"

namespace cudaq {

/// @brief Return the quantum circuit simulator for qubits.
inline nvqir::CircuitSimulator *get_simulator() {
  auto &provider = RuntimeBackendProvider::getSingleton();
  return provider.getSimulator();
}

} // namespace cudaq
