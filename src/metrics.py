def calculate_model_complexity(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_all_mb = (param_size + buffer_size) / 1024**2
    return trainable_params, size_all_mb

def estimate_mflops(model, input_shape):
    # Dummy approximation based on trainable parameters
    # In practice, use ptflops or thop library
    trainable_params, _ = calculate_model_complexity(model)
    mflops = (trainable_params * 2) / 1e6
    return mflops
