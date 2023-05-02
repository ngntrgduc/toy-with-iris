import time

def timing(func):
    def wrapper():
        tic = time.time()
        func()
        print(f'Function {func.__name__} took {(time.time()-tic)} sec to run')
    return wrapper

# @timing
# def hi():
#     for _ in range(10):
#         print(_)
#     pass

# hi()