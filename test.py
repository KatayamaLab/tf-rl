

from multiprocessing import Process, Queue
import gym


def f(q,i):
    a = gym.make('CartPole-v1')
    a.reset()
    for j in range(10000):
        s,r,d,_=a.step(a.action_space.sample())
        #q.put([s,r,d,i])
        if d:
            a.reset()


if __name__ == '__main__':

    q = Queue()

    ps = []
    #for i in range(10):
    for i in range(4):
        p = Process(target=f, args=(q,i))
        p.start()
        ps.append(p)

    a = gym.make('CartPole-v1')
    a.reset()
    for j in range(10000):
        s,r,d,_=a.step(a.action_space.sample())
        a.render()
        if d:
            a.reset()

    p.join()





def worker():
    """thread worker function"""
    print('Worker')

    a = gym.make('CartPole-v1')
    a.reset()
    for j in range(1000):
        s,r,d,_=a.step(a.action_space.sample())
        if d:
            a.reset()
        else:
            pass
            #a.render()
    return

threads = []
for i in range(5):
    t = threading.Thread(target=worker)
    threads.append(t)
    t.start()

    for j in range(1000):
        s,r,d,_=a.step(a.action_space.sample())
        if d:
            a.reset()
        else:
            pass
