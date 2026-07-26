from pathlib import Path
import torch
import torch.nn.functional as F
from stlm.tokenizer import Tokenizer
from stlm.model import GPT
from stlm.config import ModelConfig


def load_checkpoint(
    chkpt_path: str | Path, device: str = "cpu"
) -> tuple[GPT, Tokenizer]:
    chkpt = torch.load(chkpt_path, map_location=device, weights_only=False)
    model_cfg = ModelConfig(**chkpt["config"]["model"])
    model = GPT(model_cfg)
    model.load_state_dict(chkpt["model"])
    tok = Tokenizer.load(Path(chkpt["tokenizer_path"]))
    model.to(device)
    return model, tok


@torch.no_grad()
def generate(
    model: GPT, tok: Tokenizer, prompt, *, max_tokens, temperature=0.8, top_k=None
) -> str:
    model.eval()
    device = next(model.parameters()).device
    block_size = model.config.block_size
    context = tok.encode(prompt)
    for _ in range(max_tokens):
        idx = torch.tensor([context[-block_size:]], dtype=torch.long, device=device)
        logits, _ = model(idx)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            v, ix = torch.topk(logits, k)
            logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        context.append(int(next_id.item()))
    return tok.decode(context)


if __name__ == "__main__":
    model, tok = load_checkpoint("checkpoints/toy_cpu/ckpt_5000.pt", "cpu")
    print(
        generate(
            model, tok, "Shakespeare wrote:", max_tokens=500, temperature=0.8, top_k=40
        )
    )
