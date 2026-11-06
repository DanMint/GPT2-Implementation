import tiktoken
import torch
from GPT_config import GPTConfig

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