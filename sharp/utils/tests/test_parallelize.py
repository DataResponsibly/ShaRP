from sharp.utils._parallelize import parallel_loop


def test_parallel_loop():
    def square(x):
        return x * x

    iterable = range(10)

    results = parallel_loop(square, iterable, n_jobs=2, progress_bar=False)
    assert results == [x * x for x in iterable]

    results = parallel_loop(
        square, iterable, n_jobs=2, progress_bar=True, description="Test"
    )
    assert results == [x * x for x in iterable]
