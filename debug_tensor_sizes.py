def debug_tensor_shapes(model, test_data, projection, config):
    """Debug tensor shapes to understand the dimension mismatch"""
    
    # Get a sample
    idx = torch.randint(len(test_data), (1,))
    print(f"Sample index: {idx}")
    
    inputs_embeds = test_data.encodings['inputs_embeds'][idx]
    labels = test_data.encodings['labels'][idx]
    mask_label = test_data.encodings['mask_label'][idx]
    
    print(f"Original inputs_embeds shape: {inputs_embeds.shape}")
    print(f"Original labels shape: {labels.shape}")
    print(f"Original mask_label shape: {mask_label.shape}")
    
    with torch.no_grad():
        inputs_projected = projection(inputs_embeds.to(device).float())
        print(f"Projected inputs shape: {inputs_projected.shape}")
        
        sample_output = model(inputs_embeds=inputs_projected)
        print(f"Model output logits shape: {sample_output.logits.shape}")
    
    return inputs_embeds, labels, mask_label, sample_output 