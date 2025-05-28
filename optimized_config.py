from transformers import BertConfig

# Memory-optimized configuration
def get_optimized_config():
    config = BertConfig()
    
    # Model Parameter (same as original)
    config.hidden_size = 256
    config.num_hidden_layers = 4
    config.num_attention_heads = 4
    config.num_hidden_ffn = 4
    config.attention_type = "default"
    config.k = 20
    config.fixed = False
    config.add_alibi = False
    config.alibi_type = "nosym"
    config.fixed_slopes = False
    config.add_urpe = False

    config.vocab_size = 256
    config.intermediate_size = config.hidden_size * config.num_hidden_ffn
    config.max_length = 20
    config.max_position_embeddings = config.max_length
    config.position_embedding_type = 'sincos'
    config.input_type = 'trace'
    config.embedding_type = 'none'
    config.type_vocab_size = 2
    config.output_hidden_states = True
    config.output_attentions = True
    config.output_scores = True
    config.pre_ln = True

    # OPTIMIZED Training Parameters
    config.batch_size = 32  # Reduced from 256
    config.gradient_accumulation_steps = 8  # Effective batch size = 32 * 8 = 256
    config.lr = 5e-4
    config.epoch = 1000
    config.patience = 20
    
    # Mixed precision training
    config.use_amp = True
    
    # Memory optimization
    config.dataloader_num_workers = 2
    config.pin_memory = True
    config.prefetch_factor = 2

    # I/O parameter
    config.parent_dir = './results/pretrain/'
    config.dataset = './data/pretrain/'
    
    return config 