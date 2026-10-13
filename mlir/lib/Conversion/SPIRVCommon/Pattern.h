//===- Pattern.h - SPIRV Common Conversion Patterns -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SPIRVCOMMON_PATTERN_H
#define MLIR_CONVERSION_SPIRVCOMMON_PATTERN_H

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOpTraits.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
namespace spirv {

/// Converts elementwise unary, binary and ternary standard operations to SPIR-V
/// operations.
template <typename Op, typename SPIRVOp>
struct ElementwiseOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(adaptor.getOperands().size() <= 3);
    Type dstType = this->getTypeConverter()->convertType(op.getType());
    if (!dstType) {
      return rewriter.notifyMatchFailure(
          op->getLoc(),
          llvm::formatv("failed to convert type {0} for SPIR-V", op.getType()));
    }

    SmallVector<Value> operands(adaptor.getOperands());
    // Emulated narrow integers may carry any upper bits; unsigned operations
    // and logical right shifts must read only the source width.
    constexpr bool readsUnsigned =
        SPIRVOp::template hasTrait<OpTrait::spirv::UnsignedOp>() ||
        std::is_same_v<SPIRVOp, spirv::ShiftRightLogicalOp>;
    Type srcElementType = getElementTypeOrSelf(op.getType());
    auto srcIntType = dyn_cast<IntegerType>(srcElementType);
    auto dstIntType = dyn_cast<IntegerType>(getElementTypeOrSelf(dstType));
    if (readsUnsigned && !srcElementType.isIndex() && dstType != op.getType() &&
        (!srcIntType || !dstIntType ||
         srcIntType.getWidth() > dstIntType.getWidth())) {
      return op.emitError("bitwidth emulation is not implemented yet on "
                          "unsigned op pattern version");
    }
    if (readsUnsigned && srcIntType && dstIntType &&
        srcIntType.getWidth() < dstIntType.getWidth()) {
      TypedAttr maskAttr = IntegerAttr::get(
          dstIntType,
          APInt::getLowBitsSet(dstIntType.getWidth(), srcIntType.getWidth()));
      if (auto vectorType = dyn_cast<VectorType>(dstType))
        maskAttr = SplatElementsAttr::get(vectorType, maskAttr);
      Value mask =
          spirv::ConstantOp::create(rewriter, op.getLoc(), dstType, maskAttr);
      // A logical shift reads its shift amount whole.
      size_t maskedOperands =
          std::is_same_v<SPIRVOp, spirv::ShiftRightLogicalOp> ? 1
                                                              : operands.size();
      for (Value &operand :
           MutableArrayRef(operands).take_front(maskedOperands)) {
        operand = spirv::BitwiseAndOp::create(rewriter, op.getLoc(), dstType,
                                              operand, mask);
      }
    }
    rewriter.template replaceOpWithNewOp<SPIRVOp>(op, dstType, operands);
    return success();
  }
};

} // namespace spirv
} // namespace mlir

#endif // MLIR_CONVERSION_SPIRVCOMMON_PATTERN_H
