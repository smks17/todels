def model_tester(func):
    def inner():
        try:
            func()
        except Exception as e:
            assert False, f"{func.__name__} raised an exception '{e}'"
    return inner

MODEL_INPUT_SIZE = (3, 224, 224)
