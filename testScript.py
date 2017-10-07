from myo import init, Hub, Feed
from sklearn.tree import DecisionTreeClassifier
import time
import math
from statistics import mode
import numpy

init()
feed = Feed()
hub = Hub()
hub.run(1000, feed)

try:
  myo = feed.wait_for_single_device(timeout=2.0)
  if not myo:
    print("No Myo connected after 2 seconds")
  print("Hello, Myo!")
  X = []
  y = []
  # while hub.running and myo.connected:
  # print('Orientation:', quat.x, quat.y, quat.z, quat.w)

  print "Set hands !!!"
  time.sleep(3)

  time1 = time.time()
  while hub.running and myo.connected:
    # print "Reading 1"
    quat = myo.orientation
    x_q = math.ceil(quat.x * 100000) / 100000
    y_q = math.ceil(quat.y * 100000) / 100000
    z_q = math.ceil(quat.z * 100000) / 100000
    w_q = math.ceil(quat.w * 100000) / 100000
    # X.append([quat.x, quat.y, quat.z, quat.w])
    X.append([x_q, y_q, z_q, w_q])
    y.append(1)
    # print('Reading 1:', quat.x, quat.y, quat.z, quat.w)
    print('Reading 1:', x_q, y_q, z_q, w_q)
    time2 = time.time()
    if (time2 - time1) > 15:
      break

  print "Change hands !!!"
  time.sleep(3)

  time1 = time.time()
  while hub.running and myo.connected:
    # print "Reading 1"
    quat = myo.orientation
    x_q = math.ceil(quat.x * 100000) / 100000
    y_q = math.ceil(quat.y * 100000) / 100000
    z_q = math.ceil(quat.z * 100000) / 100000
    w_q = math.ceil(quat.w * 100000) / 100000
    # X.append([quat.x, quat.y, quat.z, quat.w])
    X.append([x_q, y_q, z_q, w_q])
    y.append(0)
    # print('Reading 1:', quat.x, quat.y, quat.z, quat.w)
    print('Reading 0:', x_q, y_q, z_q, w_q)
    time2 = time.time()
    if (time2 - time1) > 15:
      break

  print "Collected Data !!!"

  # Machine Learning Magic !!!
  clf = DecisionTreeClassifier(random_state=0)
  clf.fit(X,y)

  print "Change hands !!!"
  while hub.running and myo.connected:
    time.sleep(1)
    time1 = time.time()
    X = []
    # print "Reading 1"
    quat = myo.orientation
    x_q = math.ceil(quat.x * 100000) / 100000
    y_q = math.ceil(quat.y * 100000) / 100000
    z_q = math.ceil(quat.z * 100000) / 100000
    w_q = math.ceil(quat.w * 100000) / 100000
    # X.append([quat.x, quat.y, quat.z, quat.w])
    X.append([x_q, y_q, z_q, w_q])
    y1 = clf.predict(X)
    print "Result : ", y1
    # print('Reading 1:', quat.x, quat.y, quat.z, quat.w)
    # print('Test_data:', x_q, y_q, z_q, w_q)
    '''
    time2 = time.time()
    if (time2 - time1) > 0.25:
      # print "Testing now !!!!!"
      print "Result : ", numpy.mean(clf.predict(X))
      # print "Result : ", numpy.mean(y1)
      break
    '''

  # print "X = ", X
  # print "y = ", y
finally:
  hub.shutdown()  # !! crucial
