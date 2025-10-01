import iree.runtime as ireert
import numpy as np
import iree.turbine.aot as aot
import torch
import torch.nn as nn

GPU_ARCH='gfx942'
BATCH = 8
NUM_HEADS = 32
M = 32
K1 = 128
K2 = 64
N = 128

class SDPA(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, q, k, v, attn_mask):
    return nn.functional.scaled_dot_product_attention(q, k, v, attn_mask)

sdpa = SDPA()

q = torch.rand(BATCH, NUM_HEADS, M, K1, dtype = torch.float32)
k = torch.rand(BATCH, NUM_HEADS, K2, K1, dtype = torch.float32)
v = torch.rand(BATCH, NUM_HEADS, K2, N, dtype = torch.float32)
attn_mask = torch.rand(BATCH, 1, M, K2, dtype = torch.float32)
example_args = (q, k, v, attn_mask)

BATCH = torch.export.Dim("batch")
NUM_HEADS = torch.export.Dim("num_heads")
M = torch.export.Dim("M")
K2 = torch.export.Dim("K2")
dynamic_shapes = {"q": {}, "k": {2: K2}, "v": {2: K2}, "attn_mask": {3 : K2}}

# Export the program using the simple API.
export_output = aot.export(sdpa, args=example_args, dynamic_shapes=dynamic_shapes) # MLIR
export_output.save_mlir("sdpa.mlir")

# OUTPUT: 
#module @module {
#  func.func @main(%arg0: !torch.vtensor<[8,32,1,128],f32>, %arg1: !torch.vtensor<[8,32,?,128],f32>, %arg2: !torch.vtensor<[8,32,?,128],f32>, %arg3: !torch.vtensor<[8,1,1,?],f32>) -> !torch.vtensor<[8,32,1,128],f32> attributes {torch.assume_strict_symbolic_shapes} {
#    %float0.000000e00 = torch.constant.float 0.000000e+00
#    %false = torch.constant.bool false
#    %none = torch.constant.none
#    %false_0 = torch.constant.bool false
#    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2, %arg3, %float0.000000e00, %false, %none, %false_0 : !torch.vtensor<[8,32,1,128],f32>, !torch.vtensor<[8,32,?,128],f32>, !torch.vtensor<[8,32,?,128],f32>, !torch.vtensor<[8,1,1,?],f32>, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[8,32,1,128],f32>
#    return %0 : !torch.vtensor<[8,32,1,128],f32>
#  }
#}


# Compile to a deployable artifact.
export_output.session.set_flags('--iree-hal-target-device=hip')
export_output.session.set_flags('--iree-hip-target=' + GPU_ARCH)
binary = export_output.compile(save_to=None, target_backends=['rocm']) # VM FlatBuffer

# Use the IREE runtime API to test the compiled program.
config = ireert.Config('hip')
vm_module = ireert.load_vm_module(
    ireert.VmModule.copy_buffer(config.vm_instance, binary.map_memory()),
    config,
)

result = vm_module.main(q.numpy(), k.numpy(), v.numpy(), attn_mask.numpy())
print(result.to_host())
