from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3
from panda3d.core import HeightfieldTesselator

class World(ShowBase):

	def __init__(self, terrain="default.pnm"):
		ShowBase.__init__(self)

		tesselator = HeightfieldTesselator("Terrain")
		tesselator.setHeightfield(terrain)
		tesselator.setHorizontalScale(1.0)
		tesselator.setVerticalScale(20.0)

		node = tesselator.generate()

		node.reparentTo(self.render)

world = World()
world.run()
