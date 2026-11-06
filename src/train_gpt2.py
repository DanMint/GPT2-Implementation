import torch
import torch.nn as nn
from torch.nn import functional as F

from GPT_config import GPTConfig
from block import Block
from dataLoader import get_x_y

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # token embedding. converting each token id into a vector of 768 numbers
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # positional embedding. converting each position into a a vector of 768 numbers
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # B, T, C -> Batch = number of independent seqeunces being processed at the same time. T = number of tokens in each sequence. Channel = number of dimensions
        B, T = idx.size()
        assert T <= self.config.block_size 
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        # using broadcasting to add corresponding positional embeddings to each token embedding
        x = tok_emb + pos_emb

        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)

        # apply layer norm to normalize the values of a tokens vector
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            # cross entorpy does NOT like multi dimensional inputs. It cant handle (B, T, vocab_size) inputs. Here we are flattening out the 3 dimenional tensor into just 2 dimensions.
            # logits.view(-1, logits.size(-1)) -> B*T and vocab_size
            # targets.view(-1) -> this is 2D but we flatten it out
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    # loading the weights 
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

def main():
    # auto detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_avilable():
        device = "mps"
    print(f"using device: {device}")

    num_return_sequences = 5
    max_length = 30

    x, y = get_x_y()

    model = GPT(GPTConfig())
    # put model in eval model for using the model
    model.eval()
    # move everything into gpu
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    logits, loss = model(x, y)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    for i in range(50):
        # always remember to start with zero gradient!!!!
        optimizer.zero_grad()
        logits, loss = model(x, y)
        # deposits gradients += to what the gradeints are. Basically we accumulate the gradient from the loss
        loss.backward()
        # the step updates the parameters and decreases the loss
        optimizer.step()
        print(f"step {i}, loss: {loss.item()}")

    print("didnt crash yay!")
    import sys; sys.exit(0)

    # creating prefix tokens
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode("Hello, I'm a language model, ")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to('cuda')
    # here we generate
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            # take the last col logits
            logits = logits[:, -1, :]
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # topk sampling of 50 (hugging face default). We never sample rare tokens
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decode = enc.decode(tokens)
        print(">", decode)

if __name__ == "__main__":
    main()
