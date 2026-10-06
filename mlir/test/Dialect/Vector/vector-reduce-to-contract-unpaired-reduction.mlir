// RUN: mlir-opt %s --transform-interpreter | FileCheck %s

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2)>

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL: func.func @contract_broadcast_would_unpair_a_reduction
// CHECK: %[[BCAST:.+]] = vector.broadcast %arg0 : vector<3x4xf32> to vector<1x3x4xf32>
// CHECK: vector.contract
// CHECK-SAME: indexing_maps = [#[[$MAP0]], #[[$MAP0]], #[[$MAP1]]]
// CHECK-SAME: iterator_types = ["reduction", "reduction", "parallel"]
// CHECK-SAME: %[[BCAST]], %arg1, %arg2 : vector<1x3x4xf32>, vector<1x3x4xf32> into vector<4xf32>
func.func @contract_broadcast_would_unpair_a_reduction(%arg0: vector<3x4xf32>, %arg1: vector<1x3x4xf32>, %arg2: vector<4xf32>) -> vector<4xf32> {
  %bcast = vector.broadcast %arg0 : vector<3x4xf32> to vector<1x3x4xf32>
  %result = vector.contract {
    indexing_maps = [#map0, #map1, #map2],
    iterator_types = ["reduction", "reduction", "parallel"],
    kind = #vector.kind<add>
  } %bcast, %arg1, %arg2 : vector<1x3x4xf32>, vector<1x3x4xf32> into vector<4xf32>
  return %result : vector<4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %module : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f {
      transform.apply_patterns.vector.reduction_to_contract
    } : !transform.any_op
    transform.yield
  }
}
