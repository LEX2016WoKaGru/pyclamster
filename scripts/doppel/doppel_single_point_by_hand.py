import pyclamster
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

logging.basicConfig(level=logging.DEBUG)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
image_dir = os.path.join(BASE_DIR, "examples", "images", "lex")
data_dir = os.path.join(BASE_DIR, "data")

#session = pickle.load(open('data/sessions/FE3_session_new.pk','rb'))
sessions = []
sessions.append(pickle.load(open(os.path.join(data_dir,'sessions/FE3_session_new.pk'),'rb')))
sessions.append(pickle.load(open(os.path.join(data_dir,'sessions/FE4_session_new.pk'),'rb')))

#session.set_images('/home/yann/Studium/LEX/LEX/kite/cam3/FE3*.jpg')
sessions[0].set_images(os.path.join(image_dir,'cam3/FE3_Image_20160901_100000_UTCp1.jpg'))
sessions[1].set_images(os.path.join(image_dir,'cam4/FE4_Image_20160901_100000_UTCp1.jpg'))

image_list = []
for s in sessions:
    for image in s.iterate_over_rectified_images():
        image_list.append(image)

p1 = sessions[0].position
p2 = sessions[1].position

pos1 = np.array([p1.x,p1.y,p1.z])
pos2 = np.array([p2.x,p2.y,p2.z])

aziele_long = [
[4.7725572315397002],
[0.47799603447804473],
[4.7460840877307078],
[0.50654134347101099]]

azi1 = aziele_long[0][0]
ele1 = aziele_long[1][0]
azi2 = aziele_long[2][0]
ele2 = aziele_long[3][0]

#azi1 = 4.815470
#azi2 = 4.74463
#ele1 = 0.498952
#ele2 = 0.52538

aziele1=pyclamster.Coordinates3d(
    azimuth=azi1,
    elevation=ele1,
    azimuth_offset=image_list[0].coordinates.azimuth_offset,
    azimuth_clockwise=image_list[0].coordinates.azimuth_clockwise,
    elevation_type=image_list[0].coordinates.elevation_type)

aziele2=pyclamster.Coordinates3d(
    azimuth=azi2,
    elevation=ele2,
    azimuth_offset=image_list[1].coordinates.azimuth_offset,
    azimuth_clockwise=image_list[1].coordinates.azimuth_clockwise,
    elevation_type=image_list[0].coordinates.elevation_type)

doppel_raw,var_list_raw = pyclamster.doppelanschnitt(azi1,azi2,ele1,ele2,pos1,pos2,plot_info=True)
pyclamster.doppelanschnitt_plot('raw single point by hand',doppel_raw,var_list_raw,p1,p2,plot_view=1,plot_position=1)

doppel_c3d,var_list_c3d = pyclamster.doppelanschnitt_Coordinates3d(aziele1,aziele2,p1,p2,plot_info=True)
pyclamster.doppelanschnitt_plot('c3d single point by hand',doppel_c3d,var_list_c3d,p1,p2,plot_view=1,plot_position=1)

plt.show()
