import asyncio

from eidos_agents.concurrency import BoundedSpecialistExecutor


def test_parallelism_is_bounded():
    active = 0
    peak = 0

    async def work(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return value

    async def run():
        executor = BoundedSpecialistExecutor(max_parallel=2, max_calls=4, max_retries=1)
        jobs = {str(i): (lambda i=i: work(i)) for i in range(4)}
        return await executor.run(jobs)

    result = asyncio.run(run())
    assert peak == 2
    assert result == {"0": 0, "1": 1, "2": 2, "3": 3}


def test_scientific_failure_is_not_retried():
    calls = 0

    async def scientific_fail():
        nonlocal calls
        calls += 1
        raise ValueError("experiment failed")

    async def run():
        executor = BoundedSpecialistExecutor(max_parallel=1, max_calls=1, max_retries=3)
        await executor.run({"curie": scientific_fail})

    try:
        asyncio.run(run())
    except ValueError:
        pass
    assert calls == 1
