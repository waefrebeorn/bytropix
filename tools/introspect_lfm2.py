import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "D:/models/LFM2.5-2.6B"
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32, trust_remote_code=True)
model.eval()

info = {}
mm = model.model
info["embed"] = str(mm.embed_tokens.weight.shape) + " " + str(mm.embed_tokens.weight.dtype)
info["embedding_norm"] = str(mm.embedding_norm.weight.shape) + " " + str(mm.embedding_norm.weight.dtype)
info["final_norm"] = str(mm.norm.weight.shape) if hasattr(mm, "norm") else "n/a"

# layer 0 = conv, layer 2 = full_attention
for idx in [0, 2]:
    blk = mm.layers[idx]
    d = {}
    for name, mod in blk.named_children():
        if hasattr(mod, "weight"):
            d[name] = str(tuple(mod.weight.shape)) + " " + str(mod.weight.dtype)
        else:
            d[name] = type(mod).__name__
    info[f"layer{idx}"] = d

# specifically the conv block structure of layer 0
conv = mm.layers[0]
info["layer0_conv_keys"] = list(conv.conv.state_dict().keys())
cw = conv.conv.weight
info["conv_w_shape"] = str(tuple(cw.shape)) + " " + str(cw.dtype)
info["conv_groups"] = conv.conv.groups
# operator_norm / ffn_norm
info["layer0_op_norm"] = str(tuple(conv.operator_norm.weight.shape)) + " " + str(conv.operator_norm.weight.dtype)
info["layer0_ffn_norm"] = str(tuple(conv.ffn_norm.weight.shape)) + " " + str(conv.ffn_norm.weight.dtype)

# attention layer 2
att = mm.layers[2]
info["layer2_attn_keys"] = list(att.self_attn.state_dict().keys())
for k in att.self_attn.state_dict():
    t = att.self_attn.state_dict()[k]
    if hasattr(t, "shape"):
        info[f"layer2_{k}"] = str(tuple(t.shape)) + " " + str(t.dtype)

info["layer_types"] = [type(l).__name__ for l in mm.layers]
print(json.dumps(info, indent=2))
