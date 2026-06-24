from proof.rng_null_streams import suite_streams, os_urandom_bytes, system_random_digits


def test_generators_emit_valid_values():
    for spec in suite_streams("full", 42):
        gen = spec.factory()
        vals = [next(gen) for _ in range(100)]
        assert all(0 <= v < spec.size for v in vals)


def test_deterministic_generators_reproduce_with_seed():
    for spec in suite_streams("controls", 7):
        a = spec.factory(); b = spec.factory()
        assert [next(a) for _ in range(50)] == [next(b) for _ in range(50)]
        assert spec.reproducible is True


def test_os_security_sources_marked_non_reproducible():
    assert os_urandom_bytes(1).reproducible is False
    assert system_random_digits(1).reproducible is False
