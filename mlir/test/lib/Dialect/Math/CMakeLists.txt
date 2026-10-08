# Exclude tests from libMLIR.so
add_mlir_library(MLIRMathTestPasses
  TestAlgebraicSimplification.cpp
  TestPolynomialApproximation.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRMathTestPasses PUBLIC
  MLIRMathTransforms
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRX86Dialect
  )
