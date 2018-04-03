import cv2
import numpy as np
import tensorflow as tf
#from matplotlib import pyplot as plt

'''
#capture image
camera = cv2.VideoCapture(0)

while True:
	ret, frame = camera.read()
	cv2.imshow('test', frame)
	key = cv2.waitKey(3)

	#ESC
	if key%256 == 27:
		print ("Terminated")
		sys.exit(0)
	#SPACE
	elif key%256 == 32:
		print ("Frame Captured")
		cv2.imwrite("capture.jpg", frame)
		break

camera.release()
'''

#load image and template
image = cv2.imread('nine_grid.jpg', 0)
template = cv2.imread('phone_grid.jpg', 0)
height = template.shape[0]
width = template.shape[1]

#template matching
#using cv2.TM_CCOEFF
match = cv2.matchTemplate(image, template, 0)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

top_left = min_loc
bottom_right = (top_left[0] + width, top_left[1] + height)

cv2.rectangle(image, top_left, bottom_right, 255, 2)

'''
#plot result
plt.subplot(121), plt.imshow(match, cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(image, cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.show()
'''

#crop into cells
y_t = int(top_left[0]) 
x_t = int(top_left[1])
y_b = int(bottom_right[0])
x_b = int(bottom_right[1])
hor_1 = int(x_t + (x_b - x_t) / 3)
hor_2 = int(hor_1 + (x_b - hor_1) / 2)
vert_1 = int(y_t + (y_b - y_t) / 3)
vert_2 = int(vert_1 + (y_b - vert_1) / 2)

cell1 = image[x_t : hor_1, y_t : vert_1]
cell2 = image[hor_1 : hor_2, y_t : vert_1]
cell3 = image[hor_2 : x_b, y_t : vert_1]
cell4 = image[x_t : hor_1, vert_1 : vert_2]
cell5 = image[hor_1 : hor_2, vert_1 : vert_2]
cell6 = image[hor_2 : x_b, vert_1 : vert_2]
cell7 = image[x_t : hor_1, vert_2 : y_b]
cell8 = image[hor_1 : hor_2, vert_2 : y_b]
cell9 = image[hor_2 : x_b, vert_2 : y_b]
cells = [cell1, cell2, cell3, cell4, cell5, cell6, cell7, cell8, cell9]

'''
#show cells
for cell in cells:
	cv2.imshow('cell', cell)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#prepare data
np_cells = np.empty([9, 28, 28], dtype = np.float32)
for i in range(len(cells)):
	cells[i] = cv2.resize(cells[i], None, fx = 0.34, fy = 0.34, interpolation = cv2.INTER_AREA)
	cells[i] = cells[i].reshape((32, 32))
	cells[i] = cells[i][2:30, 2:30]
	cells[i] = 255 - cells[i]
	np_cells[i] = cells[i]
np_cells = np_cells.reshape(9, 28, 28, 1)

#SAVE IMAGES
#for i in range(9):
#	cv2.imwrite(str(i) + '.jpg', np_cells[i])

#Convolutional Neural Network
#model
def cnn_model_fn(features, labels, mode):
	# Input Layer
	input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

	# Convolutional Layer 1
	conv1 = tf.layers.conv2d(
		inputs = input_layer,
		filters = 32,
		kernel_size = [5, 5],
		padding = "same",
		activation = tf.nn.relu
	)

	# Pooling Layer 1
	pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2, 2], strides = 2)

	# Convolutional Layer 2
	conv2 = tf.layers.conv2d(
		inputs = pool1,
		filters = 64,
		kernel_size = [5, 5],
		padding = "same",
		activation = tf.nn.relu
	)

	# Pooling Layer 2
	pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2, 2], strides = 2)
	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    
	# Dense Layer
	dense = tf.layers.dense(inputs = pool2_flat, units = 1024, activation = tf.nn.relu)
    
	# Dropout Layer
	dropout = tf.layers.dropout(inputs = dense, rate = 0.4, training = (mode == tf.estimator.ModeKeys.TRAIN))

	# Logits Layer
	logits = tf.layers.dense(inputs = dropout, units = 10)

	# Generate predictions (for PREDICT and EVAL mode)
	# Add `softmax_tensor` to the graph. It is used for PREDICT and by the`logging_hook`
	predictions = {
		"classes": tf.argmax(input = logits, axis = 1),
		"probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
	}
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

	# Training Optimizer (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
		train_op = optimizer.minimize(loss = loss, global_step = tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

#create classifier from checkpoint directory
classifier = tf.estimator.Estimator(model_fn = cnn_model_fn, model_dir = './tf_checkpoints')

#prediction input function
pred_in_fn = tf.estimator.inputs.numpy_input_fn(
	x = {"x": np_cells},
	shuffle = False
)

#EstimatorSpec generator object
classifications = classifier.predict(input_fn = pred_in_fn)

#get the results
results = []
for i in range(9):
	results.append(next(classifications)['classes'])
print (results)