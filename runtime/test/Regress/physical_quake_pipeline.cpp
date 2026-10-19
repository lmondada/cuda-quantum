/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

// RUN: physical_quake_pipeline 2>&1 | FileCheck %s

// Tests for the built-in physical-quake lowering handler. Inferred pipelines
// are planned through the production built-in factory and stage composition
// (getPassPipeline); every pipeline produced is then actually run over the
// embedded kernel with cudaq-opt, so the checks inspect real lowered IR.

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using HandlerVector =
    std::vector<std::unique_ptr<cudaq_internal::compiler::LoweringHandler>>;

// A kernel covering the profile options: a controlled X (needing basis
// decomposition when not native) and a veq measurement (expanded by
// expand-measurements).
static const char *inputKernel = R"MLIR(
module {
  func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %q = quake.alloca !quake.veq<2>
    %r0 = quake.extract_ref %q[0] : (!quake.veq<2>) -> !quake.ref
    %r1 = quake.extract_ref %q[1] : (!quake.veq<2>) -> !quake.ref
    quake.h %r0 : (!quake.ref) -> ()
    quake.t %r1 : (!quake.ref) -> ()
    quake.x [%r1] %r0 : (!quake.ref, !quake.ref) -> ()
    %meas = quake.mz %q : (!quake.veq<2>) -> !cc.sequence<!quake.measure>
    return
  }
}
)MLIR";

/// Pipelines planned so far, checked at the end for provider-named mappings.
static std::vector<std::string> plannedPipelines;

/// Plan the pipeline of \p target through the built-in lowering handlers.
static std::string planPipeline(const cudaq::CompileTarget &target,
                                const cudaq::CompileOptions &options) {
  HandlerVector handlers =
      cudaq_internal::compiler::makeBuiltinLoweringHandlers(
          options.emulate, target.codegen.emissionFormat);
  std::string diagnostic;
  auto pipeline = cudaq_internal::compiler::getPassPipeline(
      target, options, handlers, diagnostic);
  std::string result =
      pipeline ? std::move(*pipeline) : "FAILED: " + diagnostic;
  plannedPipelines.push_back(result);
  return result;
}

/// A target whose single stage lowers to \p profile by handler inference.
static cudaq::CompileTarget
physicalTarget(const cudaq::config::PhysicalQuakeProfile &profile,
               const char *emissionFormat) {
  cudaq::CompileTarget target;
  target.codegen.emissionFormat = emissionFormat;
  target.stages.push_back({.irOutputProfile = profile,
                           .lowering = cudaq::config::DefaultLowering{}});
  return target;
}

/// Run \p pipeline over the embedded kernel with cudaq-opt; the lowered IR
/// goes straight to this process's stdout for FileCheck.
static void printLoweredIR(const std::string &pipeline) {
  if (pipeline.starts_with("FAILED: "))
    return; // No pipeline was produced to lower with.
  std::cout.flush();
  std::string command =
      "cudaq-opt -pass-pipeline='builtin.module(" + pipeline + ")' 2>&1";
  FILE *stream = popen(command.c_str(), "w");
  if (!stream) {
    std::cout << "could not invoke cudaq-opt\n";
    return;
  }
  fwrite(inputKernel, 1, std::strlen(inputKernel), stream);
  if (pclose(stream) != 0)
    std::cout << "cudaq-opt pipeline failed\n";
  std::cout.flush();
}

// All gates of the embedded kernel are native to the full example basis, and
// `any` leaves the veq measurement untouched. The adaptive-QIR early-exit
// setting is captured from the codegen context.
void test_full_basis_any() {
  cudaq::config::PhysicalQuakeProfile profile;
  profile.basisGates = {"h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"};
  std::cout << "case1: "
            << planPipeline(
                   physicalTarget(profile, "qir-adaptive:0.1:int_computations"),
                   {})
            << std::endl;
  printLoweredIR(plannedPipelines.back());
}
// CHECK: case1: hw-jit-prep-pipeline{allow-early-exit=true},
// CHECK-SAME: jit-deploy-pipeline,jit-finalize-pipeline{lower-device-calls
// CHECK-SAME: =true},basis-conversion{basis=h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK: quake.x [{{.*}}] {{.*}} : (!quake.ref, !quake.ref) -> ()
// CHECK: quake.mz {{.*}} : (!quake.veq<2>) -> !cc.sequence<!quake.measure>
// CHECK-NOT: cudaq-opt pipeline failed

// `expanded` runs expand-measurements: the veq measurement is replaced by one
// measurement per qubit.
void test_full_basis_expanded() {
  cudaq::config::PhysicalQuakeProfile profile;
  profile.basisGates = {"h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"};
  profile.measurements = cudaq::config::MeasurementForm::expanded;
  std::cout << "case2: "
            << planPipeline(physicalTarget(profile, "qir-base"), {})
            << std::endl;
  printLoweredIR(plannedPipelines.back());
}
// CHECK: case2: hw-jit-prep-pipeline{allow-early-exit=false},expand-
// CHECK-SAME: measurements,jit-deploy-pipeline,jit-finalize-pipeline{lower-
// CHECK-SAME: device-calls=true},basis-conversion
// CHECK-SAME: {basis=h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK-NOT: quake.mz {{.*}} : (!quake.veq<2>) -> !cc.sequence<!quake.measure>
// CHECK: quake.mz {{.*}} : (!quake.ref) -> !quake.measure
// CHECK: quake.discriminate {{.*}} : (!quake.measure) -> i1
// CHECK-NOT: quake.mz {{.*}} : (!quake.veq<2>) -> !cc.sequence<!quake.measure>
// CHECK-NOT: cudaq-opt pipeline failed

// Same input, restricted basis: the controlled X is decomposed into h/z(1).
void test_restricted_basis() {
  cudaq::config::PhysicalQuakeProfile profile;
  profile.basisGates = {"h", "t", "z(1)"};
  profile.measurements = cudaq::config::MeasurementForm::expanded;
  std::cout << "case3: "
            << planPipeline(physicalTarget(profile, "qir-base"), {})
            << std::endl;
  printLoweredIR(plannedPipelines.back());
}
// CHECK: case3: hw-jit-prep-pipeline{allow-early-exit=false},expand-
// CHECK-SAME: measurements,jit-deploy-pipeline,jit-finalize-pipeline{lower-
// CHECK-SAME: device-calls=true},basis-conversion{basis=h,t,z(1)}
// CHECK-NOT: quake.x
// CHECK: quake.z [{{.*}}] {{.*}} : (!quake.ref, !quake.ref) -> ()
// CHECK-NOT: cudaq-opt pipeline failed

// An empty basis imposes no gate-basis restriction: no basis-conversion in
// the pipeline. The nop codegen and emulation settings are captured; running
// the pipeline proves those emul/nop variants lower the kernel (loop
// preserving, gates untouched).
void test_unconstrained_basis() {
  cudaq::config::PhysicalQuakeProfile profile;
  profile.measurements = cudaq::config::MeasurementForm::expanded;
  cudaq::CompileOptions options;
  options.emulate = true;
  std::cout << "case4: "
            << planPipeline(physicalTarget(profile, "nop"), options)
            << std::endl;
  printLoweredIR(plannedPipelines.back());
}
// CHECK: case4: emul-jit-prep-pipeline{erase-noise=true allow-early-exit
// CHECK-SAME: =false no-loop-unroll=true},expand-measurements,jit-deploy-
// CHECK-SAME: pipeline{no-loop-unroll=true},jit-finalize-pipeline{lower-
// CHECK-SAME: device-calls=true}
// CHECK-NOT: basis-conversion
// CHECK: quake.x [{{.*}}] {{.*}} : (!quake.ref, !quake.ref) -> ()
// CHECK: cc.loop while
// CHECK: quake.mz {{.*}} : (!quake.ref) -> !quake.measure
// CHECK-NOT: cudaq-opt pipeline failed

// A declared physical stage followed by an inferred physical-to-physical
// stage: the handler lowers the second stage from the first stage's output
// profile, changing the basis. The first stage's declared profile matches
// its actual `canonicalize` output (any-measurement, full basis), so the
// inferred stage truly starts from physical-quake input.
void test_physical_to_physical() {
  cudaq::config::PhysicalQuakeProfile full, restricted;
  full.basisGates = {"h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"};
  restricted.basisGates = {"h", "t", "z(1)"};
  restricted.measurements = cudaq::config::MeasurementForm::expanded;
  cudaq::CompileTarget target;
  target.codegen.emissionFormat = "qir-base";
  cudaq::config::CustomPipelineLowering custom;
  custom.overridePassPipeline = "canonicalize";
  target.stages.push_back({.irOutputProfile = full, .lowering = custom});
  target.stages.push_back({.irOutputProfile = restricted,
                           .lowering = cudaq::config::DefaultLowering{}});
  std::cout << "case5: " << planPipeline(target, {}) << std::endl;
  printLoweredIR(plannedPipelines.back());
}
// CHECK: case5: canonicalize,hw-jit-prep-pipeline{allow-early-exit
// CHECK-SAME: =false},expand-measurements,jit-deploy-pipeline,jit-finalize-
// CHECK-SAME: pipeline{lower-device-calls=true},basis-conversion
// CHECK-SAME: {basis=h,t,z(1)}
// CHECK: quake.z [{{.*}}] {{.*}} : (!quake.ref, !quake.ref) -> ()
// CHECK-NOT: cudaq-opt pipeline failed

// reset-before-reuse requests are rejected until WP8 adds the lowering: the
// composition fails and the reason is reported.
void test_reset_rejected() {
  cudaq::config::PhysicalQuakeProfile profile;
  profile.basisGates = {"h", "t", "z(1)"};
  profile.resetBeforeReuse = true;
  std::string pipeline = planPipeline(physicalTarget(profile, "qir-base"), {});
  std::cout << "case6: " << pipeline << std::endl;
}
// CHECK: profile requests reset-before-reuse lowering, which is not
// CHECK-SAME: supported yet
// CHECK: case6: FAILED: compile target stage 1: no lowering handler
// CHECK-SAME: supports lowering from unchecked to physical-quake

// No inferred pipeline contains provider-named gate-set mappings.
void test_no_provider_mappings() {
  for (const auto &pipeline : plannedPipelines)
    for (const char *mapping :
         {"ionq-gate-set-mapping", "fermioniq-gate-set-mapping",
          "quantinuum-gate-set-mapping"})
      if (pipeline.find(mapping) != std::string::npos) {
        std::cout << "unexpected provider mapping in: " << pipeline << "\n";
        return;
      }
  std::cout << "case7: no provider-named mappings\n";
}
// CHECK: case7: no provider-named mappings

int main() {
  test_full_basis_any();
  test_full_basis_expanded();
  test_restricted_basis();
  test_unconstrained_basis();
  test_physical_to_physical();
  test_reset_rejected();
  test_no_provider_mappings();
  return 0;
}
