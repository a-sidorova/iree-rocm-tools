import iree.runtime as ireert
import numpy as np
import iree.turbine.aot as aot
import torch
import torch.nn as nn

GPU_ARCH='gfx942'


class SDPA(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, q, k, v):
    return nn.functional.scaled_dot_product_attention(q, k, v)

sdpa = SDPA()

q = torch.rand(1, 16, 128, 64, dtype = torch.float32)
k = torch.rand(1, 16, 128, 64, dtype = torch.float32)
v = torch.rand(1, 16, 128, 64, dtype = torch.float32)

# Export the program using the simple API.
export_output = aot.export(sdpa, q, k, v) # MLIR
export_output.save_mlir("sdpa.mlir")

# OUTPUT: 
#module @module {
#  func.func @main(%arg0: !torch.vtensor<[1,16,128,64],f32>, %arg1: !torch.vtensor<[1,16,128,64],f32>, %arg2: !torch.vtensor<[1,16,128,64],f32>) -> !torch.vtensor<[1,16,128,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
#    %none = torch.constant.none
#    %float0.000000e00 = torch.constant.float 0.000000e+00
#    %false = torch.constant.bool false
#    %none_0 = torch.constant.none
#    %false_1 = torch.constant.bool false
#    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2, %none, %float0.000000e00, %false, %none_0, %false_1 : !torch.vtensor<[1,16,128,64],f32>, !torch.vtensor<[1,16,128,64],f32>, !torch.vtensor<[1,16,128,64],f32>, !torch.none, !torch.float, !torch.bool, !torch.none, !torch.bool -> !torch.vtensor<[1,16,128,64],f32>
#    return %0 : !torch.vtensor<[1,16,128,64],f32>
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

result = vm_module.main(q.numpy(), k.numpy(), v.numpy())
print(result.to_host())
