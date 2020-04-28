import numpy as np
import tensorflow as tf 
import PFNNLayer as PFNN
from PFNNLayer import Layer
import os.path


tf.random.set_seed(23456)  # reproducibility


""" Load Data """

database = np.load('database.npz')
X = database['Xun']
Y = database['Yun']
P = database['Pun']

# (N, 342) (N, 311)
# (450, 258) (450, 221)
print(X.shape, Y.shape)

""" Calculate Mean and Std """

Xmean, Xstd = X.mean(axis=0), X.std(axis=0)
Ymean, Ystd = Y.mean(axis=0), Y.std(axis=0)

# print(Xstd)

j = 21
w = ((60*2)//10)

Xstd[w*0:w* 1] = Xstd[w*0:w* 1].mean() # Trajectory Past Positions
Xstd[w*1:w* 2] = Xstd[w*1:w* 2].mean() # Trajectory Future Positions
Xstd[w*2:w* 3] = Xstd[w*2:w* 3].mean() # Trajectory Past Directions
Xstd[w*3:w* 4] = Xstd[w*3:w* 4].mean() # Trajectory Future Directions
Xstd[w*4:w*10] = Xstd[w*4:w*10].mean() # Trajectory Gait

""" Mask Out Unused Joints in Input """

joint_weights = np.array([
  1, 
  1, 1, 1, 1, # 1e-10, 1e-10,
  1, 1e-10, 1, 1, # 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10,
  1, 1e-10, 1, 1, # 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10,
  1e-10, 1, 1, 1,
  1e-10, 1, 1, 1 ]).repeat(3)

"""
np.array([
    1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
    1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10]).repeat(3)
    """

Xstd[w*10+j*3*0:w*10+j*3*1] = Xstd[w*10+j*3*0:w*10+j*3*1].mean() / (joint_weights * 0.1) # Pos
Xstd[w*10+j*3*1:w*10+j*3*2] = Xstd[w*10+j*3*1:w*10+j*3*2].mean() / (joint_weights * 0.1) # Vel
Xstd[w*10+j*3*2:          ] = Xstd[w*10+j*3*2:          ].mean() # Terrain

Ystd[0:2] = Ystd[0:2].mean() # Translational Velocity
Ystd[2:3] = Ystd[2:3].mean() # Rotational Velocity
Ystd[3:4] = Ystd[3:4].mean() # Change in Phase
Ystd[4:8] = Ystd[4:8].mean() # Contacts

Ystd[8+w*0:8+w*1] = Ystd[8+w*0:8+w*1].mean() # Trajectory Future Positions
Ystd[8+w*1:8+w*2] = Ystd[8+w*1:8+w*2].mean() # Trajectory Future Directions

Ystd[8+w*2+j*3*0:8+w*2+j*3*1] = Ystd[8+w*2+j*3*0:8+w*2+j*3*1].mean() # Pos
Ystd[8+w*2+j*3*1:8+w*2+j*3*2] = Ystd[8+w*2+j*3*1:8+w*2+j*3*2].mean() # Vel
Ystd[8+w*2+j*3*2:8+w*2+j*3*3] = Ystd[8+w*2+j*3*2:8+w*2+j*3*3].mean() # Rot

""" Save Mean / Std / Min / Max """
Xmean.tofile('./data/Xmean.bin')
Ymean.tofile('./data/Ymean.bin')
Xstd.tofile('./data/Xstd.bin')
Ystd.tofile('./data/Ystd.bin')


""" Normalize Data """
X = (X - Xmean) / Xstd
Y = (Y - Ymean) / Ystd
P = P[:,np.newaxis]


input_size = X.shape[1]+1 # we input both X and P, hence +1 (N, 343)
output_size = Y.shape[1] # 331

"""data for training"""
input_x = np.concatenate((X,P),axis = 1) #input of nn, including X and P
input_y = Y


N =input_x.shape[0]
print("Data is processed")

#--------------------------------above is dataprocess-------------------------------------




""" Phase Function Neural Network """


"""input"""
tf.compat.v1.disable_eager_execution()
X_nn = tf.compat.v1.placeholder(tf.float32, [None, input_size], name='x-input')
Y_nn = tf.compat.v1.placeholder(tf.float32, [None, output_size], name='y-input')
# initialize tensorw - X_nn (? * 343) Y_nn (? * 331) whose row# is unknown


"""parameter"""

rng = np.random.RandomState(23456)
nslices = 4                             
phase = X_nn[:,-1] # column with phase
P0 = Layer((nslices, 512, input_size-1), rng, phase, 'wb0')
P1 = Layer((nslices, 512, 512), rng, phase, 'wb1')
P2 = Layer((nslices, output_size, 512), rng, phase, 'wb2')

"""structure"""

# Input Layer
H0 = tf.expand_dims(X_nn[:,:-1], -1)# (? * 342 * 1) - built from input shape excluding phase
H0 = tf.nn.dropout(H0, rate = 0.3) 

b0 = tf.expand_dims(P0.bias, -1) # (? * 512 * 1) which is just P0.bias

# Hidden Layer 1
H1 = tf.matmul(P0.weight, H0) + b0 # (? * 512 * 342) mul (? * 342 * 1) = (? * 512 * 1)
H1 = tf.nn.elu(H1) # P0 encoded weights from H0 to H1 and bias
H1 = tf.nn.dropout(H1, rate = 0.3)

b1 = tf.expand_dims(P1.bias, -1) # (? * 512 * 1)

# Hiddeln Layer 2
H2 = tf.matmul(P1.weight, H1) + b1 # (? * 512 * 512) mul (? * 512 * 1) = (? * 512 * 1)
H2 = tf.nn.elu(H2) # ELU activation
H2 = tf.nn.dropout(H2, rate = 0.3)

b2 = tf.expand_dims(P2.bias, -1) # ( ? * 311 * 1)

# Output Layer
H3 = tf.matmul(P2.weight, H2) + b2 # (? * 311 * 521) mul (? * 512 * 1) = (? * 311 * 1)
H3 = tf.compat.v1.squeeze(H3, -1) # (? * 311 * 1)


"""loss function and trainer"""

# minimize gamma times the variation of the abs of all weights
def regularization_penalty(a0, a1, a2, gamma):
    return gamma * (tf.reduce_mean(tf.abs(a0))+tf.reduce_mean(tf.abs(a1))+tf.reduce_mean(tf.abs(a2)))/3
    # returns a constant

# cost between Y and output
cost = tf.reduce_mean(tf.square(Y_nn - H3))
loss = cost + regularization_penalty(P0.alpha_W, P1.alpha_W, P1.alpha_W, 0.01)

#optimizer, learning rate 0.0001
learning_rate = 0.0001
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) 
# the "trainer“ that minimize a loss function


"""session"""

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

#saver for saving the variables
saver = tf.compat.v1.train.Saver()


#start to train
print('Learning start..')
batch_size = 32
training_epochs = 3
total_batch = int(N / batch_size)
print("totoal_batch:", total_batch)
# use random state to shuffle order or examples
I = np.arange(N) 
rng.shuffle(I)
error = np.ones(training_epochs)

for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        # take the window of one batch from shuffled example
        index_train = I[i * batch_size : (i + 1) * batch_size]

        batch_xs = input_x[index_train]
        batch_ys = input_y[index_train]

        # feed training data into placeholder
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys}
        l, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += l / total_batch
        
        if i % 10 == 0:
            print(i, "loss:", l)
    
    save_path = saver.save(sess, "./model/model.ckpt")
    PFNN.save_network((sess.run(P0.alpha_W), sess.run(P1.alpha_W), sess.run(P2.alpha_W)), 
                      (sess.run(P0.alpha_b), sess.run(P1.alpha_b), sess.run(P2.alpha_b)), 
                      nslices,
                      50, 
                      '')

    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.9f}'.format(avg_cost))
    error[epoch] = avg_cost
    error.tofile("./model/error.bin")
    
    
#-----------------------------above is model training----------------------------------
