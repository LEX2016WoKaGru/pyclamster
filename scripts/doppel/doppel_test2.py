import pyclamster
import pickle
import matplotlib.pyplot as plt
import gc
import numpy as np
import os

def on_click(time,image,image_left_selected,event):
    x = int(event.xdata)
    y = int(event.ydata)
    azi = image.coordinates.azimuth[y,x]
    ele = image.coordinates.elevation[y,x]
    print('---- pixel-pos = ('+str(x)+','+str(y)+') ----')
    print('left_image = '+str(image_left_selected[-1]))
    print('time = '+str(time))
    print('azi  = '+str(azi))
    print('ele  = '+str(ele))
    if image_left_selected[-1]:
        marked_left_azi.append(azi)
        marked_left_ele.append(ele)
        image_left_selected.append(False)
        print('### select matching point in RIGHT image')
    else:
        marked_right_azi.append(azi)
        marked_right_ele.append(ele)
        image_left_selected.append(True)
        print('### select new pair of pixel (first in the LEFT image)')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
image_dir = os.path.join(BASE_DIR, "examples", "images", "lex")
data_dir = os.path.join(BASE_DIR, "data")

#session = pickle.load(open('data/sessions/FE3_session_new.pk','rb'))
sessions = []
sessions.append(pickle.load(open(os.path.join(data_dir,'sessions/FE3_session_new.pk'),'rb')))
sessions.append(pickle.load(open(os.path.join(data_dir,'sessions/FE4_session_new.pk'),'rb')))

#session.set_images('/home/yann/Studium/LEX/LEX/kite/cam3/FE3*.jpg')
sessions[0].set_images(os.path.join(image_dir,'cam3/FE3_Image_20160901_103000_UTCp1.jpg'))
sessions[1].set_images(os.path.join(image_dir,'cam4/FE4_Image_20160901_103000_UTCp1.jpg'))

marked_left_azi = []
marked_left_ele = []
marked_right_azi = []
marked_right_ele = []
image_list = []
# loop over all images
for s in sessions:
    image_list.append([])
    for image in s.iterate_over_rectified_images():
        image_list[-1].append(image)

for i in [0]:
    image_left_selected = [True]
    time = image_list[1][i]._get_time_from_filename("FE4_Image_%Y%m%d_%H%M%S_UTCp1.jpg")
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(image_list[0][i].data)
    plt.title("cam3 "+str(time))
    plt.axis('off')
    ax2 = fig.add_subplot(122)
    ax2.imshow(image_list[1][i].data)
    plt.title("cam4 "+str(time))
    plt.axis('off')
    fig.canvas.mpl_connect('scroll_event',lambda e:on_click(time,image,image_left_selected,e))
    plt.show()
    #    print("{} times recorded".format(len(session_time)))
    #del image
    #gc.collect()

aziele3=pyclamster.Coordinates3d(
    azimuth=marked_left_azi,
    elevation=marked_left_ele,
    azimuth_offset=image_list[0][0].coordinates.azimuth_offset,
    azimuth_clockwise=image_list[0][0].coordinates.azimuth_clockwise,
    elevation_type=image_list[0][0].coordinates.elevation_type)

aziele4=pyclamster.Coordinates3d(
    azimuth=marked_right_azi,
    elevation=marked_right_ele,
    azimuth_offset=image_list[1][0].coordinates.azimuth_offset,
    azimuth_clockwise=image_list[1][0].coordinates.azimuth_clockwise,
    elevation_type=image_list[1][0].coordinates.elevation_type)

doppel = pyclamster.positioning.doppelanschnitt_Coordinates3d(
    aziele1 = aziele3,
    aziele2 = aziele4,
    pos1    = sessions[0].position,
    pos2    = sessions[1].position
    )

ax = doppel.plot3d()
ax.scatter3D(sessions[0].position.x,sessions[0].position.y,sessions[0].position.z,c='r')
ax.scatter3D(sessions[1].position.x,sessions[1].position.y,sessions[1].position.z,c='g')
plt.show()


#pickle.dump([session_coord,np.unique(session_time)[:-1]],
#    open("data/cam_kite/FE4_cam_kite_new.pk","wb"))

