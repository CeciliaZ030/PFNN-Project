from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3
from pandac.PandaModules import HeightfieldTesselator

class Display(ShowBase):

	def __init__(self):
		Showbase.__init__(self)

		self.scene
