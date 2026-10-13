; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 < %s | FileCheck %s

; A one-element f64 vector keeps its vector type through the combine.

; CHECK-LABEL: {{^}}copysign_v1f64:
; CHECK: v_bfi_b32
; CHECK: global_store_dwordx2
define amdgpu_kernel void @copysign_v1f64(ptr addrspace(1) %in, ptr addrspace(1) %out) {
  %x = load <1 x double>, ptr addrspace(1) %in
  %nonzero = fcmp one <1 x double> %x, zeroinitializer
  %magnitude = uitofp <1 x i1> %nonzero to <1 x double>
  %result = call <1 x double> @llvm.copysign.v1f64(<1 x double> %magnitude, <1 x double> %x)
  store <1 x double> %result, ptr addrspace(1) %out
  ret void
}

declare <1 x double> @llvm.copysign.v1f64(<1 x double>, <1 x double>)
