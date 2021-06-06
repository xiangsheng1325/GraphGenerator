import time

def time_decorator(func):
    def time_record(*args, **kwargs):
        startt = time.time()
        res = func(*args, **kwargs)
        endt = time.time()
        time_consumption = endt - startt
        print("Time Consumption of {}: {:.6f}s.".format(func.__name__, time_consumption))
        return res
    return time_record


@time_decorator
def test_deco(n):
    for i in range(n):
        continue


if __name__ == '__main__':
    n = 1048576
    test_deco(n)
    for i in range(7):
        n *= 2
        test_deco(n)