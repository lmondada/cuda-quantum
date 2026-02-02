/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ResourceCounter.h"
#include "common/Logger.h"
#include "common/RuntimeBackendProvider.h"
#include "nvqir/CircuitSimulator.h"

namespace nvqir {

void setChoiceFunction(std::function<bool()> choice) {
  auto &provider = cudaq::RuntimeBackendProvider::getSingleton();
  if (!provider.getResourceCounterSimulator()) {
    CUDAQ_WARN("SimulatorType is not ResourceCounterSimulator, ignoring choice "
               "function");
    return;
  }
  provider.getResourceCounterSimulator()->setChoiceFunction(choice);
}

cudaq::Resources *getResourceCounts() {
  auto &provider = cudaq::RuntimeBackendProvider::getSingleton();
  if (!provider.getResourceCounterSimulator()) {
    CUDAQ_WARN("SimulatorType is not ResourceCounterSimulator, no resource "
               "counts available");
    return nullptr;
  }
  provider.getResourceCounterSimulator()->flushGateQueue();
  return provider.getResourceCounterSimulator()->getResourceCounts();
}
} // namespace nvqir

NVQIR_REGISTER_SIMULATOR(nvqir::ResourceCounter, resource_counter)
