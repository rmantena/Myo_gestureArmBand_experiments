# Example added by Rajiv Mantena
# rmantena.github.io
#
# This code could be used/modified/published by anyone, free of charge.
# Don't hold me responsible if something dosne't work as it should.
# Not perfectly tested to check if there is any stream missing, from either Myos.
#
# Tested with two Myo Bands, connected to a single Bluetooth Dongle on a PC.


from __future__ import print_function
from myo import init, Hub, Feed, StreamEmg
import time

init()
feed = Feed()
hub = Hub()
hub.run(1000, feed)

try:
    myMyos = feed.get_connected_devices()
    # Enable stream for the Myos
    for myo in myMyos:
        myo.set_stream_emg(StreamEmg.enabled)
    while hub.running:
        if myMyos:
            # Check if Myos are streaming
            if all(myo.emg is not None for myo in myMyos):
                # Save to a list and print in a neat format
                emgStream = []
                print("EMG Data Scream : ", end="")
                for myo in myMyos:
                    emgStream.append(myo.emg)
                for a in emgStream:
                    for b in a:
                        print('{:4}'.format(b), end="")
                print("")
            time.sleep(0.1)

finally:
    hub.shutdown()