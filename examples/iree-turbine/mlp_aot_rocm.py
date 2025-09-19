import iree.runtime as ireert
import numpy as np
import iree.turbine.aot as aot
import torch
import torch.nn as nn

GPU_ARCH='gfx942'


class MLP(nn.Module):
  def __init__(self, in_features, hidden_features, out_features):
    super().__init__()
    self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )

  def forward(self, input):
    return self.layers(input)


in_features = 256
hidden_features = 128
out_features = 64
mlp = MLP(in_features, hidden_features, out_features)

# Export the program using the simple API.
export_output = aot.export(mlp, torch.randn(in_features)) # MLIR

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

input = np.array(torch.randn(in_features), dtype=np.float32)
result = vm_module.main(input)
print(result.to_host())
