import torch
from typing import Optional, Tuple

class Head(torch.nn.Module):

    def __init__(self, head_size: int, block_size:int, embedding_size: int, dropout:int, device: torch.device):
        super().__init__()

        self.key = torch.nn.Linear(embedding_size, head_size, bias=False)
        self.query = torch.nn.Linear(embedding_size, head_size, bias=False)
        self.value = torch.nn.Linear(embedding_size, head_size, bias=False)
        self.mask = torch.tril(torch.ones(size=[block_size, block_size], dtype=torch.float)).to(device)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        QK = q @ k.transpose(-2, -1) * C ** -0.5
        attention = QK.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attention = torch.nn.functional.softmax(input=attention, dim=-1)
        
        out = attention @ v
        return out
    
class MultiHeadAttention(torch.nn.Module):

    def __init__(self, num_heads: int, head_size: int, block_size:int, embedding_size:int, dropout:int, device: torch.device):
        super().__init__()
        self.heads = torch.nn.ModuleList([Head(head_size, block_size, embedding_size, dropout=dropout, device=device) for _ in range(num_heads)])
        self.projection = torch.nn.Linear(head_size * num_heads, embedding_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.projection(out))
        return out
    
class FeedForward(torch.nn.Module):

    def __init__(self, embedding_size:int, dropout:int ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(embedding_size, 4 * embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * embedding_size, embedding_size),
            torch.nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
class Block(torch.nn.Module):

    def __init__(self, embedding_size:int, num_heads: int, block_size:int, dropout: int, device: torch.device):
        super().__init__()
        head_size = embedding_size // num_heads
        self.self_attention = MultiHeadAttention(num_heads=num_heads, head_size=head_size, block_size=block_size, embedding_size=embedding_size, dropout=dropout, device=device)
        self.feed_forward = FeedForward(embedding_size=embedding_size, dropout=dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(embedding_size)
        self.layer_norm_2 = torch.nn.LayerNorm(embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))

        return x
    
class GPTModel(torch.nn.Module):

    def __init__(self, vocab_size:int, embedding_size:int, num_heads: int, block_size:int, num_layers:int, dropout:int, device: torch.device):
        super().__init__()

        self.block_size = block_size
        self.token_embedding_table = torch.nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = torch.nn.Embedding(block_size, embedding_size)
        self.device = device

        self.blocks = torch.nn.Sequential(
            *[Block(embedding_size, num_heads, block_size, dropout, device=device) for _ in range(num_layers)]
        )

        self.final_layer_norm = torch.nn.LayerNorm(embedding_size)
        self.final_linear_layer = torch.nn.Linear(embedding_size, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module: torch.nn.Module) -> None:

        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens: torch.Tensor, target: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, T = input_tokens.shape
        token_embedding = self.token_embedding_table(input_tokens)
        position_embedding = self.position_embedding_table(torch.arange(T, device=self.device))
        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.final_linear_layer(x)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, target)

        return logits, loss
    
    def generate(self, input_tokens: torch.Tensor, max_new_tokens: int) -> torch.Tensor:

        for _ in range(max_new_tokens):
            cropped_input = input_tokens[:, -self.block_size]
            logits, _ = self(cropped_input)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim = 1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_tokens = torch.cat((input_tokens, idx_next), dim=1)
        return input_tokens
    
def print_model(model: torch.nn.Module, i = '') -> None:

    for name, child in model.named_children():
        param = [p.numel() for p in child.parameters()]
        print(f"{i}|_ {name}:{child.__class__.__name__}, ({param} parameters)")
        print_model(child, i="|  ")
