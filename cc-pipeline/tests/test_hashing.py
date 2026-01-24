from cc_pipeline.common.hashing import sha256_hex, stable_url_hash


def test_sha256_hex_deterministic():
    a = sha256_hex(b"hello")
    b = sha256_hex(b"hello")
    c = sha256_hex(b"hello!")
    assert a == b
    assert a != c
    assert len(a) == 64


def test_stable_url_hash_length_and_stability():
    url = "https://example.com/a/b?x=1"
    h1 = stable_url_hash(url)
    h2 = stable_url_hash(url)
    assert h1 == h2
    assert len(h1) == 24