def model_tester(func):
    def inner():
        try:
            func()
        except Exception as e:
            assert False, f"{func.__name__} raised an exception '{e}'"
    return inner
