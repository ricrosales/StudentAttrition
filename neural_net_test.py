from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from matplotlib.pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
from numpy.random import multivariate_normal
from pybrain.datasets import SupervisedDataSet
import pandas
from array import array

#Getting data
def build_df(csv_location):
    db = pandas.read_csv(csv_location)
    db.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)
    return db

df = build_df('Trial_Data(100).csv')
labels = df.pop('exit_status').values
features = df.values

feature_array = array('f')
features = features.tolist()
feature_array.fromlist(features)
features = feature_array

labels = labels.tolist()
label_array = array('c')
label_array.fromlist(labels)
labels = label_array

alldata = SupervisedDataSet(len(features[0]), len(labels[0]))
assert labels.shape[0] == features.shape[0]
alldata.setField('input', features)
alldata.setField('target', labels)

#From:
#http://pybrain.org/docs/tutorial/fnn.html

#To have a nice dataset for visualization, we produce a set of
# points in 2D belonging to three different classes. You could
# also read in your data from a file, e.g. using pylab.load().
# means = [(-1,0),(2,4),(3,1)]
# cov = [diag([1,1]), diag([0.5,1.2]), diag([1.5,0.7])]
# alldata = ClassificationDataSet(2, 1, nb_classes=3)
# for n in xrange(400):
#     for klass in range(3):
#         input = multivariate_normal(means[klass],cov[klass])
#         alldata.addSample(input, [klass])


#Randomly split the dataset into 75% training and 25% test
# data sets. Of course, we could also have created two different
# datasets to begin with.
tstdata, trndata = alldata.splitWithProportion( 0.25 )

#For neural network classification, it is highly advisable
# to encode classes with one output neuron per class.
# Note that this operation duplicates the original targets
# and stores them in an (integer) field named 'class'
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

#Test our dataset by printing a little information about it.
print "Number of training patterns: ", len(trndata)
print "Input and output dimensions: ", trndata.indim, trndata.outdim
print "First sample (input, target, class):"
print trndata['input'][0], trndata['target'][0], trndata['class'][0]
print alldata

#Now build a feed-forward network with 5 hidden units.
# We use the shortcut buildNetwork() for this. The input and
# output layer size must match the dataset's input and target
# dimension. You could add additional hidden layers by inserting
# more numbers giving the desired layer sizes.

# The output layer uses a softmax function because we are doing classification.
# There are more options to explore here, e.g. try changing the
# hidden layer transfer function to linear instead of (the default) sigmoid.
fnn = buildNetwork( trndata.indim, 5, trndata.outdim, outclass=SoftmaxLayer )

#Set up a trainer that basically takes the network and training dataset
# as input. For a list of trainers, see trainers. We are using a BackpropTrainer for this.
trainer = BackpropTrainer( fnn, dataset=trndata, momentum=0.1, verbose=True, weightdecay=0.01)

#Now generate a square grid of data points and put it into a dataset,
# which we can then classify to obtain a nice contour field for visualization.
# Therefore the target values for this data set can be ignored.
ticks = arange(-3.,6.,0.2)
X, Y = meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=3)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

#Start the training iterations.
for i in range(20):
    #Train the network for some epochs. Usually you would set something like 5 here,
    # but for visualization purposes we do this one epoch at a time.
    trainer.trainEpochs( 1 )

    #Evaluate the network on the training and test data. There are several
    # ways to do this - check out the pybrain.tools.validation module, for instance.
    # Here we let the trainer do the test.
    trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
    tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult

    #Run our grid data through the FNN, get the most likely
    #class and shape it into a square array again.
    out = fnn.activateOnDataset(griddata)
    out = out.argmax(axis=1)  # the highest output activation gives the class
    out = out.reshape(X.shape)

    #Now plot the test data and the underlying grid as a filled contour.
    figure(1)
    ioff()  # interactive graphics off
    clf()   # clear the plot
    hold(True) # overplot on
    for c in [0,1,2]:
        here, _ = where(tstdata['class']==c)
        plot(tstdata['input'][here,0],tstdata['input'][here,1],'o')
    if out.max()!=out.min():  # safety check against flat field
        contourf(X, Y, out)   # plot the contour
    ion()   # interactive graphics on
    draw()  # update the plot

    #Finally, keep showing the plot until user kills it.
    ioff()
    show()