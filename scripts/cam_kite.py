import pyclamster
import pickle
import matplotlib.pyplot as plt
import gc

session3 = pickle.load(open('data/sessions/FE3_session.pk','rb'))
session4 = pickle.load(open('data/sessions/FE4_session.pk','rb'))

session3.set_images('/home/pi/kite/cam3/FE3*.jpg')
session4.set_images('/home/pi/kite/cam4/FE4*.jpg')

session3_azi = []
session3_ele = []
session3_time = []
def on_click(time,image,event):
    print(event.xdata,event.ydata)
    x = int(event.xdata)
    y = int(event.ydata)
    session3_azi.append(image.coordinates.azimuth[y,x])
    session3_ele.append(image.coordinates.elevation[y,x])
    session3_time.append(time)
    print('azi  = '+str(session3_azi[-1]))
    print('ele  = '+str(session3_ele[-1]))
    print('time = '+str(session3_time[-1]))

# loop over all images
for image in session3.iterate_over_images():
#    plt.subplot(131)
    print('#####')
    time = image._get_time_from_filename("FE3_Image_%Y%m%d_%H%M%S_UTCp1.jpg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(image.data)
    plt.title("image "+str(time))
    fig.canvas.mpl_connect('scroll_event',lambda e:on_click(time,image,e))
#    plt.subplot(132)
#    plt.imshow(image.coordinates.elevation)
#    plt.title("elevation")
#    plt.subplot(133)
#    plt.imshow(image.coordinates.azimuth)
#    plt.title("azimuth")
    plt.show()
    del image
    gc.collect()
    

