/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_ASSIGNIDS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {
bool isMeasureOp(Operation *op) {
  return dyn_cast<quake::MxOp>(*op) || dyn_cast<quake::MyOp>(*op) ||
         dyn_cast<quake::MzOp>(*op);
}

int numClassicalInput(Operation *op) {
  if (dyn_cast<quake::RxOp>(*op) || dyn_cast<quake::RyOp>(*op) ||
      dyn_cast<quake::RzOp>(*op))
    return 1;

  if (dyn_cast<quake::PhasedRxOp>(*op))
    return 2;

  return 0;
}

class NullWirePat : public OpRewritePattern<quake::NullWireOp> {
public:
  unsigned *counter;

  NullWirePat(MLIRContext *context, unsigned *c)
      : OpRewritePattern<quake::NullWireOp>(context), counter(c) {}

  LogicalResult matchAndRewrite(quake::NullWireOp alloc,
                                PatternRewriter &rewriter) const override {
    if (alloc->hasAttr("qid"))
      return failure();

    auto qid = (*counter)++;

    rewriter.startRootUpdate(alloc);
    alloc->setAttr("qid", rewriter.getUI32IntegerAttr(qid));
    rewriter.finalizeRootUpdate(alloc);

    return success();
  }
};

std::optional<uint> findQid(Value v) {
  auto defop = v.getDefiningOp();
  if (!defop)
    return std::nullopt;

  if (defop->getRegions().size() != 0) {
    defop->emitOpError(
        "AssignIDsPass cannot handle non-function operations with regions."
        " Do you have if statements in a Base Profile QIR program?");
    return std::nullopt;
  }

  if (!isa<quake::WireType>(v.getType()))
    return std::nullopt;

  assert(quake::isLinearValueForm(defop) &&
         "AssignIDsPass requires operations to be in value form");

  if (defop->hasAttr("qid")) {
    uint qid = defop->getAttr("qid").cast<IntegerAttr>().getUInt();
    return std::optional<uint>(qid);
  }

  // Figure out matching operand
  size_t i = 0;
  for (; i < defop->getNumResults(); i++)
    if (defop->getResult(i) == v)
      break;

  // Special cases where result # != operand #:
  // Wire is second output but sole input
  if (isMeasureOp(defop))
    i = 0;
  // Classical values preceding wires as input are consumed and not part of the results
  i += numClassicalInput(defop);
  // Swap op swaps wires
  if (dyn_cast<quake::SwapOp>(defop))
    i = (i == 1 ? 0 : 1);

  return findQid(defop->getOperand(i));
}

class SinkOpPat : public OpRewritePattern<quake::SinkOp> {
public:
  SinkOpPat(MLIRContext *context) : OpRewritePattern<quake::SinkOp>(context) {}

  LogicalResult matchAndRewrite(quake::SinkOp release,
                                PatternRewriter &rewriter) const override {
    auto qid = findQid(release.getOperand());

    if (!qid.has_value())
      return failure();

    rewriter.startRootUpdate(release);
    release->setAttr("qid", rewriter.getUI32IntegerAttr(qid.value()));
    rewriter.finalizeRootUpdate(release);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct AssignIDsPass : public cudaq::opt::impl::AssignIDsBase<AssignIDsPass> {
  using AssignIDsBase::AssignIDsBase;

  void runOnOperation() override {
    auto func = getOperation();

    if (!func->hasAttr("cudaq-kernel") || func.getBlocks().empty())
      return;

    if (!func.getFunctionBody().hasOneBlock()) {
      func.emitError("AssignIDsPass cannot handle multiple blocks. Do "
                     "you have if statements in a Base Profile QIR program?");
      signalPassFailure();
      return;
    }

    assign();
  }

  void assign() {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    unsigned x = 0;
    patterns.insert<NullWirePat>(ctx, &x);
    patterns.insert<SinkOpPat>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addDynamicallyLegalOp<quake::NullWireOp>(
        [&](quake::NullWireOp alloc) { return alloc->hasAttr("qid"); });
    target.addDynamicallyLegalOp<quake::SinkOp>(
        [&](quake::SinkOp sink) { return sink->hasAttr("qid"); });
    if (failed(applyPartialConversion(func.getOperation(), target,
                                      std::move(patterns)))) {
      func.emitOpError("Assigning qids failed");
      signalPassFailure();
    }
  }
};

} // namespace