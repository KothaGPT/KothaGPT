from ml.inference.quant import dequantize_model, quantize_model
from ml.models import KothaGPT
from ml.models.config import ModelConfig
from ml.tokenizer import load_tokenizer


def test_quantize_smoke():
    tok = load_tokenizer("ml/tokenizer/artifacts/best/tokenizer.json")
    cfg = ModelConfig(vocab_size=len(tok.vocab), hidden_size=32, num_layers=1, num_heads=4, max_position_embeddings=64)
    m = KothaGPT(cfg)
    q = quantize_model(m, bits=8)
    assert hasattr(q, "_quant_bits")
    dq = dequantize_model(q)
    assert not hasattr(dq, "_quant_bits")

def test_quantize_invalid():

    from ml.models import KothaGPT
    from ml.models.config import ModelConfig
    tok_len = 698
    cfg = ModelConfig(vocab_size=tok_len, hidden_size=32, num_layers=1, num_heads=4, max_position_embeddings=64)
    m = KothaGPT(cfg)
    try:
        quantize_model(m, bits=7)
        assert False
    except ValueError:
        pass
