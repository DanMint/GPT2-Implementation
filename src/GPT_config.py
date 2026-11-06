from dataclasses import dataclass

@dataclass
class GPTConfig:
    # max sequence length
    block_size: int = 1024

    # number of tokens: 50000 BPE merges + 256 bytes tokens + <|endofline|>
    vocab_size: int = 50257

    # number of layers
    n_layer: int = 12

    # number of heads
    n_head: int = 12

    # embedding dimension
    n_embd: int = 768
    