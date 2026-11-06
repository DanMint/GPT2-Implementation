import tiktoken
import torch
from GPT_config import GPTConfig

"""
    This class implements a simple data loader.
    The main idea is that we are going through the file in chunks
"""
class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        # load tokens from input.txt and store them in memory
        with open('input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_poistion = 0
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_poistion : self.current_poistion+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)

        # advance poistion in the tensor
        self.current_poistion += B * T

        # if loading next batch is out of bounds, reset
        if self.current_poistion + (B * T + 1) > len(self.tokens):
            self.current_poistion = 0

        return x, y  


def get_x_y():
    enc = tiktoken.get_encoding('gpt2')
    with open('input.txt', 'r') as f:
        text = f.read()

    text = text[:1000]
    tokens = enc.encode(text)
    B, T = 4, 32
    buf = torch.tensor(tokens[:B*T + 1])
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)

    return x, y

def main():
    x, y = get_x_y()
    print(x)
    print(y)

if __name__ == "__main__":
    main()