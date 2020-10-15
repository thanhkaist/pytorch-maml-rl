from  torch.multiprocessing import Pool
import torch.multiprocessing as mp
import time


def foo(num,div):
    begin = time.time()
    print(str(num) + " divide for " + str(div))
    val = [i for i in range(num)]
    for i in range(len(val)):
        val[i] /= div

    print("Time: " + str(time.time() -begin)  +   str(num) + " divide for " + str(div))

    return sum(val)



if __name__ == '__main__':
    arg = [(1000,10) , (100000,11), (2000000,200)]
    proc = []
    for i in arg:
        p = mp.Process(target=foo,args=i)
        p.start()
        proc.append(p)

    for p in proc:
        p.join()

