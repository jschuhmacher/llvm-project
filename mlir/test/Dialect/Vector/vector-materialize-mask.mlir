// RUN: mlir-opt %s --transform-interpreter --split-input-file | FileCheck %s

func.func @select_single_i1_vector(%cond : i1) -> vector<1xi1> {
  %true = arith.constant dense<true> : vector<1xi1>
  %false = arith.constant dense<false> : vector<1xi1>
  %select = arith.select %cond, %true, %false : i1, vector<1xi1>
  return %select : vector<1xi1>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.materialize_masks
    } : !transform.op<"func.func">
    transform.yield
  }
}

// CHECK-LABEL: func @select_single_i1_vector
// CHECK-SAME: %[[COND:.*]]: i1
// CHECK:      %[[BCAST:.*]] = vector.broadcast %[[COND]] : i1 to vector<1xi1>
// CHECK:      return %[[BCAST]] : vector<1xi1>

// -----

func.func @masked_write_keeps_create_mask(%buf : memref<32xi4>, %v : vector<8xi4>, %i : index, %n : index) {
  %m = vector.create_mask %n : vector<8xi1>
  vector.transfer_write %v, %buf[%i], %m : vector<8xi4>, memref<32xi4>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.materialize_masks preserve_create_masks
    } : !transform.op<"func.func">
    transform.yield
  }
}

// CHECK-LABEL: func @masked_write_keeps_create_mask
// CHECK-SAME:    %[[BUF:.*]]: memref<32xi4>, %[[V:.*]]: vector<8xi4>, %[[I:.*]]: index, %[[N:.*]]: index
// CHECK-NOT:     arith.cmpi
// CHECK-NOT:     arith.andi
// CHECK:         %[[REST:.*]] = arith.subi %{{.*}}, %[[I]] : index
// CHECK:         %[[BOUND:.*]] = arith.minsi %[[REST]], %[[N]] : index
// CHECK:         %[[MASK:.*]] = vector.create_mask %[[BOUND]] : vector<8xi1>
// CHECK:         vector.transfer_write %[[V]], %[[BUF]][%[[I]]], %[[MASK]] {in_bounds = [true]}

// -----

func.func @masked_write_default_lowers(%buf : memref<32xi4>, %v : vector<8xi4>, %i : index, %n : index) {
  %m = vector.create_mask %n : vector<8xi1>
  vector.transfer_write %v, %buf[%i], %m : vector<8xi4>, memref<32xi4>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.materialize_masks
    } : !transform.op<"func.func">
    transform.yield
  }
}

// CHECK-LABEL: func @masked_write_default_lowers
// CHECK-NOT:     vector.create_mask
// CHECK:         arith.cmpi sgt
// CHECK:         arith.cmpi sgt
// CHECK:         arith.andi

// -----

func.func @and_of_create_masks_2d(%a0 : index, %a1 : index, %b0 : index, %b1 : index) -> vector<4x8xi1> {
  %a = vector.create_mask %a0, %a1 : vector<4x8xi1>
  %b = vector.create_mask %b0, %b1 : vector<4x8xi1>
  %and = arith.andi %a, %b : vector<4x8xi1>
  return %and : vector<4x8xi1>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.materialize_masks preserve_create_masks
    } : !transform.op<"func.func">
    transform.yield
  }
}

// CHECK-LABEL: func @and_of_create_masks_2d
// CHECK-SAME:    %[[A0:.*]]: index, %[[A1:.*]]: index, %[[B0:.*]]: index, %[[B1:.*]]: index
// CHECK-DAG:     %[[D0:.*]] = arith.minsi %[[A0]], %[[B0]] : index
// CHECK-DAG:     %[[D1:.*]] = arith.minsi %[[A1]], %[[B1]] : index
// CHECK:         %[[MASK:.*]] = vector.create_mask %[[D0]], %[[D1]] : vector<4x8xi1>
// CHECK-NOT:     arith.andi
// CHECK:         return %[[MASK]]

// -----

func.func @and_of_create_mask_and_constant_not_composed(%n : index) -> vector<8xi1> {
  %a = vector.create_mask %n : vector<8xi1>
  %b = arith.constant dense<[true, false, true, true, false, true, true, true]> : vector<8xi1>
  %and = arith.andi %a, %b : vector<8xi1>
  return %and : vector<8xi1>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root : !transform.any_op {transform.readonly}) {
    %func_op = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.op<"func.func">
    transform.apply_patterns to %func_op {
      transform.apply_patterns.vector.materialize_masks preserve_create_masks
    } : !transform.op<"func.func">
    transform.yield
  }
}

// CHECK-LABEL: func @and_of_create_mask_and_constant_not_composed
// CHECK:         %[[MASK:.*]] = vector.create_mask
// CHECK:         arith.andi %[[MASK]], %{{.*}} : vector<8xi1>
