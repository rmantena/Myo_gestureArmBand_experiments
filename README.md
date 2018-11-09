# Myo_gestureArmBand_experiments

## Custom gesture recognition with the Myo Armband by implementing scikit-learn machine learning models. 

This project is using the Python bindings for Myo SDK provided by Niklas Rosenstein at https://github.com/NiklasRosenstein/myo-python. Give his repo a star. 

Once the bindings are installed as instructed by Niklas, try and see if you could run one of his example python scripts (in the ./examples/ directory).

If Niklas's examples don't work, then you've done something wrong with the bindings themselves. You'll have to figure that out first. 

If that's successful, then could try running the newRunScript.py script that I've written here. 

If Niklas's examples work and mine don't, then you're still missing some python libraries like scikit-learn, numpy etc. Get those installed and try again (see stack-overflow for help). It 'should' work as long as all the libraries are in place. 

I featurized the input data stream from the Myo based on the models provided in this research paper: http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7748960
Those models helped me transform the stream of data into something more meaningful for the classifiers.

I'm a noob when it comes to deep-learning, so just implemented random-forests and svm.

This project is licensed under the MIT License. 
Use however you like it. It's provided as-is.
Copyright © 2018 Rajiv Mantena
