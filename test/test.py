import tensorflow as tf
import numpy as np

#a=tf.Variable([[1,2],[1,1]],dtype=tf.float32,name='A')
#b=tf.Variable([[2,5],[3,5]],dtype=tf.float32,name='B')
#c=tf.square(tf.add(a,b,'ADD'),'SQUARE')
#f=tf.constant([[2,2],[2,2]],dtype=tf.float32,name='F')

const=tf.constant(100,dtype=tf.float32)

p = tf.placeholder(tf.float32, [1])
w1 = tf.Variable([1,2,3],dtype=tf.float32,name='W1')
w2 = tf.Variable([3,7],dtype=tf.float32,name='W2')

t1 = tf.constant([3,4,5],dtype=tf.float32,name='T1')
t2 = tf.constant([5000,9000],dtype=tf.float32,name='T2')

y1 = w1*p
y2 = w2*p

loss1 = (y1-t1)**2
loss2 = (y2-t2)**2

#o=tf.train.GradientDescentOptimizer(0.001,name='GD')
o1=tf.train.AdamOptimizer(10.0,name='GD')
o2=tf.train.AdamOptimizer(10.0,name='GD')
m1=o1.minimize(loss1,name='MINIMIZE1')
m2=o2.minimize(loss2,name='MINIMIZE2')

#g=o.compute_gradients(c)
#h=[(k[0]*ff,k[1]) for k in g]

i=tf.global_variables_initializer()

#start session

s=tf.Session()

s.run(i)

print(s.run([y1,y2],feed_dict={p:np.array([10])} ))

for i in range(1000):
    s.run([m1,m2],feed_dict={p:np.array([10])} )

print(s.run([y1,y2,loss1],feed_dict={p:np.array([10])} ))

#log = tf.summary.FileWriter('./log', s.graph)



#summary = s1.run(log)
#log.add_summary(summary, 1)


# s1=tf.Session()
# s2=tf.Session()

# s1.run(i)
# s2.run(i)
#
# cc=s1.run(c)
# print(cc)
# print('---')
# cc=s2.run(c)
# print(cc)
# print('===')
#
# print(s1.run(g))
#
# for _ in range(3):
#
#   #s1.run(m)
#   cc=s1.run(c)
#   print(cc)
#
#   print('---')
#
#   s2.run(o.apply_gradients(h))
#   cc=s2.run(c)
#   print(cc)
#
#   print('===')
