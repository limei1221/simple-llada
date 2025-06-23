
class Config:
    def __init__(
        self,
        n_layer: int,
        n_head: int,
        n_embd: int,
        *,
        n_query_groups: int = 1,
        block_size: int = 2048,  # 2048 in SMDM paper, 4096 in LLaDA paper
        bias: bool = False,
        vocab_size: int = 32_000,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_query_groups = n_query_groups
        self.block_size = block_size
        self.bias = bias
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.norm_eps = norm_eps

        # derived
        self.head_dim = n_embd // n_head
        self.intermediate_size = 4 * n_embd

    @classmethod
    def from_name(cls, name: str) -> "Config":
        if (
            name == "LLaMA_34M"
        ):  # 25M non-embedding parameters (34M model in SMDM paper, Table 8)
            return cls(n_layer=8, n_head=8, n_embd=512, n_query_groups=1, block_size=1024)
        if (
            name == "LLaMA_85M"
        ):  # 85M non-embedding parameters (113M model in SMDM paper, Table 8)
            return cls(n_layer=12, n_head=12, n_embd=768, n_query_groups=1, block_size=1024)
        if (
            name == "LLaMA_1B"
        ):  # 946M non-embedding parameters (1B model in LLaDA paper, Table 5)
            return cls(n_layer=22, n_head=32, n_embd=2048, n_query_groups=8, block_size=4096)
        raise ValueError(name)
