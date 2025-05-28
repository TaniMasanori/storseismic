# SIMPLE OPTIMIZED REPLACEMENT CELL
# Copy and paste this to replace the original slow cell

mult_factor = 10

# Optimized data multiplication using dictionary comprehension
snist_train_mlm = {
    key: tensor.repeat(mult_factor, 1, 1) if tensor.dim() == 3 and key != 'index'
    else tensor.repeat(mult_factor) if tensor.dim() == 1 and key != 'index'
    else tensor
    for key, tensor in snist_train_mlm.items()
}
snist_train_mlm['index'] = torch.arange(snist_train_mlm['inputs_embeds'].shape[0])

snist_test_mlm = {
    key: tensor.repeat(mult_factor, 1, 1) if tensor.dim() == 3 and key != 'index'
    else tensor.repeat(mult_factor) if tensor.dim() == 1 and key != 'index'
    else tensor
    for key, tensor in snist_test_mlm.items()
}
snist_test_mlm['index'] = torch.arange(snist_test_mlm['inputs_embeds'].shape[0])

print(f"Optimized data multiplication completed (factor: {mult_factor})")
print(f"Training shape: {snist_train_mlm['inputs_embeds'].shape}")
print(f"Test shape: {snist_test_mlm['inputs_embeds'].shape}") 