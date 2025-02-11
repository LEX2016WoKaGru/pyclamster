import pyclamster
import pickle
import matplotlib.pyplot as plt
import gc
import numpy as np
import os
import logging
logging.basicConfig(level=logging.DEBUG)



def on_click(image_left,image_right,event):
    global image_left_selected, marked_left_azi, marked_left_ele, marked_right_azi, marked_right_ele
    x = int(event.xdata)
    y = int(event.ydata)
    if image_left_selected:
        azi = image_left.coordinates.azimuth[y, x]
        ele = image_left.coordinates.elevation[y, x]
        marked_left_azi.append(azi)
        marked_left_ele.append(ele)
        image_left_selected = False
        print('### select matching point in RIGHT image')
    else:
        azi = image_right.coordinates.azimuth[y, x]
        ele = image_right.coordinates.elevation[y, x]
        marked_right_azi.append(azi)
        marked_right_ele.append(ele)
        image_left_selected = True
        print('### Selected new pair')
    # print('---- pixel-pos = ('+str(x)+','+str(y)+') ----')
    # print('left_image = '+str(image_left_selected[-1]))
    # print('azi  = '+str(azi))
    # print('ele  = '+str(ele))


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
image_dir = os.path.join(BASE_DIR, "examples", "images", "lex")
data_dir = os.path.join(BASE_DIR, "data")

#session = pickle.load(open('data/sessions/FE3_session_new.pk','rb'))
sessions = []
sessions.append(pickle.load(open(os.path.join(data_dir,'sessions/FE3_session_new_600.pk'),'rb')))
sessions.append(pickle.load(open(os.path.join(data_dir,'sessions/FE4_session_new_600.pk'),'rb')))

print(sessions[0].position)
print(sessions[1].position)
#session.set_images('/home/yann/Studium/LEX/LEX/kite/cam3/FE3*.jpg')
#image_dir = '/home/tfinn/Data/specific/'
#sessions[0].set_images(os.path.join(image_dir, 'cam3'))
#sessions[1].set_images(os.path.join(image_dir, 'cam4'))
sessions[0].set_images(os.path.join(image_dir,'cam3/FE3_Image_20160901_100000_UTCp1.jpg'))
sessions[1].set_images(os.path.join(image_dir,'cam4/FE4_Image_20160901_100000_UTCp1.jpg'))


marked_left_azi = []
marked_left_ele = []
marked_right_azi = []
marked_right_ele = []
image_list = []
for s in sessions:
    for image in s.iterate_over_rectified_images():
        image_list.append(image)
image_left = image_list[0]
image_right = image_list[1]
image_left_selected = True

fig = plt.figure()
#ax1 = fig.add_subplot(221)
ax1 = fig.add_subplot(121)
ax1.imshow(image_left.data,interpolation='nearest')
plt.title("cam3")
plt.axis('off')
#ax2 = fig.add_subplot(222)
ax2 = fig.add_subplot(122)
ax2.imshow(image_right.data,interpolation='nearest')
plt.title("cam4")
plt.axis('off')
# ax1 = fig.add_subplot(223)
# ax1.imshow(image_left.coordinates.azimuth,interpolation='nearest')
# plt.title("cam3")
# plt.axis('off')
# ax2 = fig.add_subplot(224)
# ax2.imshow(image_right.coordinates.azimuth,interpolation='nearest')
# plt.title("cam4")
# plt.axis('off')
fig.canvas.mpl_connect('scroll_event',lambda e:on_click(image_left,image_right,e))
plt.show()

# plot :
azi1 = marked_left_azi
ele1 = marked_left_ele
azi2 = marked_right_azi
ele2 = marked_right_ele

p1 = sessions[0].position
p2 = sessions[1].position

aziele1=pyclamster.Coordinates3d(
    azimuth=azi1,
    elevation=ele1,
    azimuth_offset=image_list[0].coordinates.azimuth_offset,
    azimuth_clockwise=False,
    elevation_type=image_list[0].coordinates.elevation_type)

aziele2=pyclamster.Coordinates3d(
    azimuth=azi2,
    elevation=ele2,
    azimuth_offset=image_list[1].coordinates.azimuth_offset,
    azimuth_clockwise=image_list[1].coordinates.azimuth_clockwise,
    elevation_type=image_list[1].coordinates.elevation_type)
############
print(aziele1.elevation_type, aziele1.azimuth_offset, aziele1.azimuth_clockwise, aziele2.elevation_type, aziele2.azimuth_offset, aziele2.azimuth_clockwise)
doppel_c3d,var_list_c3d = pyclamster.doppelanschnitt_Coordinates3d(aziele1,aziele2,p1,p2,plot_info=True)
pyclamster.doppelanschnitt_plot('c3d single point by hand',doppel_c3d,var_list_c3d,p1,p2,plot_view=1,plot_position=1)

############
lon1,lat1 = pyclamster.positioning.Projection().xy2lonlat(p1.x,p1.y)
lon2,lat2 = pyclamster.positioning.Projection().xy2lonlat(p2.x,p2.y)
lons,lats = pyclamster.positioning.Projection().xy2lonlat(doppel_c3d.x,doppel_c3d.y)
#
# ax,m = pyclamster.positioning.plot_results2d(lons,lats,doppel_c3d.z)
# x1,y1 = m(lon1,lat1)
# x2,y2 = m(lon2,lat2)
# ax.plot([x1],[y1],'go')
# ax.plot([x2],[y2],'bo')

############

ax,m = pyclamster.positioning.plot_results3d(lons,lats,doppel_c3d.z)
x1,y1 = m(lon1,lat1)
x2,y2 = m(lon2,lat2)
ax.plot([x1],[y1],[p1.z],'gx')
ax.plot([x2],[y2],[p2.z],'bx')

plt.show()

