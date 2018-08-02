#importing tensorflow
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#to suppress deprecative messages
tf.logging.set_verbosity(tf.logging.ERROR)

#Defing the Tensorflow session variale
sess=tf.InteractiveSession()

#reading the data set  (don't know if this is only for mnist or for different
#datasets)
mnist = input_data.read_data_sets("path_to_folder_which_conatins/MNIST_data/",one_hot=True)

#designing the placeholders as we know our images are in the shape (1,784)
#placeholder  for images
input_images = tf.placeholder(tf.float32,shape=[None,784])
#placeholder for labels or targets or y-values if images are x
target_labels = tf.placeholder(tf.float32,shape=[None,10])

#designing our base network
input_layer = tf.layers.dense(inputs=input_images,units=512,activation=tf.nn.relu)
digit_weight = tf.layers.dense(inputs=input_layer,units=10)

#Next line contains the error or loss measuring function
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=digit_weight, labels=target_labels))

#Optimizer for training our model
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss_function)

#Measuring the accuracy
#Step1- checking if the outputs and labels are equal
correct = tf.equal(tf.argmax(digit_weight,1),tf.argmax(target_labels,1))
#step2- getting the mean of all the correct measurements
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

#initialising all the variables
tf.global_variables_initializer().run()

for x in range(2000):
	#This is an inbuilt methods to get batches
	batch = mnist.train.next_batch(100)
	#batch[0] has the image tensor of shape (no_of_iamges,784)
	#batch[1] has the label
	feed_dict={input_images:batch[0],target_labels:batch[1]}
	#runnning our optimizer
	#the main training part
	optimizer.run(feed_dict = feed_dict)
	if ((x+1) % 100 == 0):
		print("Training epoch" + str(x+1))
		print("Accuracy: " + str(accuracy.eval(feed_dict={input_images: mnist.test.images, target_labels: mnist.test.labels})))


#measuring the final accuracy 
feed_dict= {input_images:mnist.test.images,target_labels:mnist.test.labels}
print("Accuracy"  + str(accuracy.eval(feed_dict={input_images: mnist.test.images, target_labels: mnist.test.labels})))

