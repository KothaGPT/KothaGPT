from ml.inference.batcher import Batcher, BatchItem


class FakeEngine:
    def generate(self, prompt, max_new_tokens=3):
        # deterministic fake: yield chars of prompt
        yield from prompt[:max_new_tokens]

def test_batcher():
    e = FakeEngine()
    b = Batcher(e, max_batch=2)
    items = [BatchItem("hi", 2), BatchItem("hello", 3)]
    res = b.generate_many(items)
    assert res == [["h","i"], ["h","e","l"]]
    streamed = list(b.stream_many(items))
    assert len(streamed) == 5

