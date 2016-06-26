#!/usr/bin/env python3
import pyclamster


cmset = pyclamster.coordinates.CalculationMethodSet()

cmset.add_new_method(input={'x','y','z'},output='radius',
    func=lambda:print("radius=sqrt(x^2+y^2+z^3)"))

cmset.add_new_method(input={'x','y'},output='radiush',
    func=lambda:print("radiush=sqrt(x^2+y^2)"))

cmset.add_new_method(input={'radius','radiush'},output='z',
    func=lambda:print("z=sqrt(radius^2-radiush^2)"))

cmset.add_new_method(input={'x','y','azimuth_offset'},output='azimuth',
    func=lambda:print("fancy azimuth calculation..."))

cmset.add_new_method(input={'radiush','z'},output='elevation',
    func=lambda:print("fancy elevation calculation..."))

cmset.add_new_method(input={'azimuth','radius'},output='x',
    func=lambda:print("fancy x calculation..."))

print(cmset)

print("")

q1 = {'x'}
dcq1 = cmset.directly_calculatable_quantities(q1)
print("based on {q}, {d} can be calculated directly.".format(q=q1,d=dcq1))

q2 = {'x','y','z'}
depline2 = cmset.dependency_line(q2)
print("based on {q}, the dependency line is:\n{d}".format(q=q2,d=depline2))
depline2()


