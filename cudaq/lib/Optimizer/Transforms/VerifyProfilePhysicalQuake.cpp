/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_VERIFYPROFILEPHYSICALQUAKE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// Read-only verifier for physical-quake profile stage boundaries.
struct VerifyProfilePhysicalQuake
    : public cudaq::opt::impl::VerifyProfilePhysicalQuakeBase<
          VerifyProfilePhysicalQuake> {
  using VerifyProfilePhysicalQuakeBase::VerifyProfilePhysicalQuakeBase;

  void runOnOperation() override;
};

} // namespace

/// Whether the operation uses a value-form (SSA) quantum type.
static bool hasNonReferenceValue(Operation *op) {
  if (llvm::any_of(op->getOperandTypes(), cudaq::quake::isQuantumValueType) ||
      llvm::any_of(op->getResultTypes(), cudaq::quake::isQuantumValueType))
    return true;
  return llvm::any_of(op->getRegions(), [&](Region &region) {
    return llvm::any_of(region.getBlocks(), [&](Block &block) {
      return llvm::any_of(block.getArgumentTypes(),
                          cudaq::quake::isQuantumValueType);
    });
  });
}

/// Whether the operation takes or returns quantum references.
static bool hasQuantumRefValue(Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      cudaq::quake::isQuantumReferenceType) ||
         llvm::any_of(op->getResultTypes(),
                      cudaq::quake::isQuantumReferenceType);
}

/// Whether the operation passes an opaque C++/kernel callable,
/// which may capture or allocate quantum state in an unavailable body.
static bool hasCallableValue(Operation *op) {
  return llvm::any_of(op->getOperandTypes(), [](Type t) {
    return isa<cudaq::cc::CallableType, cudaq::cc::IndirectCallableType>(t);
  });
}

void VerifyProfilePhysicalQuake::runOnOperation() {
  auto module = getOperation();

  // Requests for checking that is not implemented yet must fail explicitly
  // rather than succeed unchecked.
  if (resetBeforeReuse) {
    module.emitError("cannot verify reset-before-reuse: reset checking is "
                     "not implemented yet");
    signalPassFailure();
    return;
  }
  if (measurements != "any" && measurements != "expanded") {
    module.emitError(
        "invalid value '" + measurements +
        "' for option 'measurements' (expected 'any' or 'expanded')");
    signalPassFailure();
    return;
  }
  const bool expandedForm = measurements == "expanded";

  // An empty basis imposes no gate-basis restriction.
  std::unique_ptr<ConversionTarget> basisTarget;
  if (!basis.empty())
    basisTarget = cudaq::createBasisTarget(*module.getContext(), basis);

  // Resolves from the call's nearest symbol table; failed, non-function,
  // or bodyless resolution is unprovable (never a crash).
  auto hasBody = [](Operation *from, SymbolRefAttr symbol) {
    auto func =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(from, symbol);
    return func && !func.isExternal();
  };

  bool failed = false;
  module.walk([&](Operation *op) {
    if (hasNonReferenceValue(op)) {
      op->emitOpError("profile violation: physical-quake requires "
                      "reference-form Quake (no !quake.wire, "
                      "!quake.control, or !quake.cable values)");
      failed = true;
    }
    if (expandedForm) {
      if (auto meas = dyn_cast<cudaq::quake::MeasurementInterface>(op)) {
        auto targets = meas.getTargets();
        if (targets.size() != 1) {
          op->emitOpError("profile violation: expanded measurement form "
                          "requires exactly one !quake.ref target (got ")
              << targets.size() << " targets)";
          failed = true;
        } else if (!isa<cudaq::quake::RefType>(targets[0].getType())) {
          op->emitOpError("profile violation: expanded measurement form "
                          "requires a !quake.ref target (got ")
              << targets[0].getType() << ")";
          failed = true;
        }
      }
    }

    if (!basisTarget && !expandedForm)
      return;

    // Applications and calls whose gates cannot be accounted for cannot be
    // proven against the basis: reject as unverifiable, never approve.
    if (auto apply = dyn_cast<cudaq::quake::ApplyOp>(op)) {
      if (basisTarget && !apply.getControls().empty()) {
        op->emitOpError("cannot verify: controlled kernel application is "
                        "not provable against the target basis");
        failed = true;
      } else if (apply.getIndirectCallee() || !apply.getCallee() ||
                 !hasBody(op, *apply.getCallee())) {
        op->emitOpError("cannot verify: applied kernel body is not available");
        failed = true;
      }
      return;
    }
    // compute_action's callable bodies are opaque to this pass.
    if (isa<cudaq::quake::ComputeActionOp>(op)) {
      op->emitOpError("cannot verify: compute_action bodies are not available");
      failed = true;
      return;
    }
    if (auto call = dyn_cast<CallOpInterface>(op)) {
      auto symbol = dyn_cast<SymbolRefAttr>(call.getCallableForCallee());
      func::FuncOp callee;
      if (symbol)
        callee = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, symbol);
      // An indirect or unresolved callee's body is statically unavailable,
      // as is a declaration-only one; accept the latter only when provably
      // classical (no quantum-reference or opaque-callable operands and
      // not declared a quantum kernel).
      bool unavailable = !callee || callee.isExternal();
      bool provablyClassical = callee && !hasQuantumRefValue(op) &&
                               !hasCallableValue(op) &&
                               !callee->hasAttr(cudaq::kernelAttrName);
      if (unavailable && !provablyClassical) {
        op->emitOpError("cannot verify: called function body is not available");
        failed = true;
      }
      return;
    }

    if (!basisTarget)
      return;

    if (!op->getDialect() ||
        !isa<cudaq::quake::QuakeDialect>(op->getDialect()) ||
        basisTarget->isLegal(op))
      return;

    // A gate with a dynamically-sized control vector cannot be proven
    // against finite-arity entries: reject as unverifiable, distinct
    // from a proven violation (name absent from the basis).
    if (auto optor = dyn_cast<cudaq::quake::OperatorInterface>(op);
        optor && !cudaq::getKnownNumControls(optor)) {
      auto name = optor->getName().stripDialect();
      if (llvm::any_of(basis, [&](const std::string &b) {
            return StringRef(b).take_front(b.find_first_of("(<")) == name;
          })) {
        op->emitOpError("cannot verify: number of controls is not "
                        "statically known and the basis has no "
                        "unbounded-arity entry for it");
        failed = true;
        return;
      }
    }
    op->emitOpError("profile violation: operation is not in the target basis");
    failed = true;
  });

  if (failed)
    signalPassFailure();
}
