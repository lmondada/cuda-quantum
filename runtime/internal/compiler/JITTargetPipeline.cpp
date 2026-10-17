/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq/Target/CompilationStage.h"
#include "cudaq/runtime/logger/logger.h"
#include <regex>
#include <variant>

static void appendPipelineStage(std::string &pipeline,
                                const std::string &stage) {
  if (stage.empty())
    return;
  if (!pipeline.empty())
    pipeline += ",";
  pipeline += stage;
}

/// Rewrite `qubit-mapping{...device=...}` to use `device=bypass`.
static void setQubitMappingBypass(std::string &pipeline) {
  std::regex qubitMapping("(.*)qubit-mapping\\{(.*)device=[^,\\}]+(.*)\\}(.*)");
  std::string replacement("$1qubit-mapping{$2device=bypass$3}$4");
  pipeline = std::regex_replace(pipeline, qubitMapping, replacement);
}

/// Append the JIT pipeline built around the user-provided pipelines of a
/// `CustomPipelineLowering` to `passPipeline`.
static void appendResolvedStagePipeline(
    std::string &passPipeline,
    const cudaq::config::CustomPipelineLowering &lowering,
    const cudaq::CompileTarget &target, const cudaq::CompileOptions &options) {
  if (!lowering.overridePassPipeline.empty()) {
    appendPipelineStage(passPipeline, lowering.overridePassPipeline);
    return;
  }

  const std::string lowerDeviceCalls =
      (target.codegen.emissionFormat == "nop" && !options.emulate) ? "false"
                                                                   : "true";
  const std::string allowEarlyExit =
      target.codegen.emissionFormat.starts_with("qir-adaptive") ? "true"
                                                                : "false";
  const std::string noLoopUnroll =
      target.codegen.emissionFormat == "nop" ? " no-loop-unroll=true" : "";

  if (passPipeline.empty())
    passPipeline = "canonicalize";

  if (options.emulate)
    appendPipelineStage(passPipeline, "emul-jit-prep-pipeline{erase-noise=true "
                                      "allow-early-exit=" +
                                          allowEarlyExit + noLoopUnroll + "}");
  else
    appendPipelineStage(passPipeline, "hw-jit-prep-pipeline{allow-early-exit=" +
                                          allowEarlyExit + noLoopUnroll + "}");

  const std::string deployStage =
      target.codegen.emissionFormat == "nop"
          ? "jit-deploy-pipeline{no-loop-unroll=true}"
          : "jit-deploy-pipeline";
  const std::string finalizeStage =
      "jit-finalize-pipeline{lower-device-calls=" + lowerDeviceCalls + "}";

  appendPipelineStage(passPipeline, lowering.highLevelPipeline);
  appendPipelineStage(passPipeline, deployStage);
  appendPipelineStage(passPipeline, lowering.midLevelPipeline);
  appendPipelineStage(passPipeline, finalizeStage);
  appendPipelineStage(passPipeline, lowering.lowLevelPipeline);

  // Handle disable_qubit_mapping runtime option.
  if (target.disableQubitMapping) {
    setQubitMappingBypass(passPipeline);
    CUDAQ_INFO("disable_qubit_mapping option found, so updated lowering "
               "pipeline to {}",
               passPipeline);
  }
}

std::string cudaq_internal::compiler::getPassPipeline(
    const cudaq::CompileTarget &target, const cudaq::CompileOptions &options) {
  std::string passPipeline;

  for (const auto &stage : target.stages)
    std::visit(
        [&](const auto &lowering) {
          appendResolvedStagePipeline(passPipeline, lowering, target, options);
        },
        stage.lowering);

  return passPipeline;
}
