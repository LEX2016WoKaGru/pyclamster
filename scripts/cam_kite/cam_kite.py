import pyclamster
import pickle
import matplotlib.pyplot as plt
import gc
import numpy as np

#session = pickle.load(open('data/sessions/FE3_session_new.pk','rb'))
session = pickle.load(open('data/sessions/FE4_session_new.pk','rb'))

#session.set_images('/home/yann/Studium/LEX/LEX/kite/cam3/FE3*.jpg')
session.set_images('/home/yann/Studium/LEX/LEX/kite/cam4/FE4*.jpg')

session_azi = []
session_ele = []
session_time = []
def on_click(time,image,event):
    print(event.xdata,event.ydata)
    x = int(event.xdata)
    y = int(event.ydata)
    session_time.append(time)
    session_azi.append(image.coordinates.azimuth[y,x])
    session_ele.append(image.coordinates.elevation[y,x])
    print('azi  = '+str(session_azi[-1]))
    print('ele  = '+str(session_ele[-1]))
    print('time = '+str(session_time[-1]))


# loop over all images
for image in session.iterate_over_images():
#    plt.subplot(131)
    print('#####')
    time = image._get_time_from_filename("FE4_Image_%Y%m%d_%H%M%S_UTCp1.jpg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image.data,interpolation="nearest")
    plt.title("image "+str(time))
    fig.canvas.mpl_connect('scroll_event',lambda e:on_click(time,image,e))
#    plt.subplot(132)
#    plt.imshow(image.coordinates.elevation)
#    plt.title("elevation")
#    plt.subplot(133)
#    plt.imshow(image.coordinates.azimuth)
#    plt.title("azimuth")
    plt.show()
    print("{} times recorded".format(len(session_time)))
    del image
    gc.collect()
    

session_coord=pyclamster.Coordinates3d(
    azimuth=session_azi,
    elevation=session_ele,
    azimuth_offset=3/2*np.pi,
    azimuth_clockwise=True,
    elevation_type="ground"
    )

pickle.dump([session_coord,session_time],
    open("data/cam_kite/FE4_cam_kite_new.pk","wb"))

