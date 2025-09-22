from transformers import AutoModel, AutoTokenizer
import iree.runtime as ireert
import iree.turbine.aot as aot
import torch

GPU_ARCH='gfx942'
TEXT = 'Today is perfect day for tennis.'
MODEL = 'google-bert/bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokens = tokenizer(TEXT, return_tensors="pt", padding=True)

example_args = (tokens["input_ids"], tokens["attention_mask"])
input_spec = (torch.export.Dim("batch", min=1, max=8), torch.export.Dim("seq", min=1, max=512))
dynamic_shapes = {"input_ids": {1: input_spec[1]}, "attention_mask": {1: input_spec[1]}}

# Attention! To successfully export the model to MLIR, we have to merge the PR 21817 in IREE
model = AutoModel.from_pretrained(MODEL)
export_output = aot.export(model, args=example_args, dynamic_shapes=dynamic_shapes) # MLIR

export_output.session.set_flags('--iree-hal-target-device=hip')
export_output.session.set_flags('--iree-hip-target=' + GPU_ARCH)
binary = export_output.compile(save_to=None, target_backends=['rocm']) # VM FlatBuffer

# Use the IREE runtime API to test the compiled program.
config = ireert.Config('hip')
vm_module = ireert.load_vm_module(
    ireert.VmModule.copy_buffer(config.vm_instance, binary.map_memory()),
    config,
)

result = vm_module.main(tokens["input_ids"].numpy(), tokens["attention_mask"].numpy())
print(result)
