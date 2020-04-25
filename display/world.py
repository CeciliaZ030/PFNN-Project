from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import DirectionalLight

from terrain import generate_terrain

class World(ShowBase):

	def __init__(self, terrain="default.png"):
		ShowBase.__init__(self)

		#Initialize scene
		base.setBackgroundColor( 0.41, 0.61, 0.93 )
		base.disableMouse()

		#Lighting
		dlight = DirectionalLight("light1")
		dlnode = self.render.attachNewNode(dlight)
		self.render.setLight(dlnode)

		#Camera
		self.camera.setPos(-1000, 1000, 500)
		self.camera.setHpr(-135, -15, 0)
		#self.camera.lookAt(0, 0, 0)

		#Terrain
		terrain = generate_terrain()
		terrain.reparentTo(self.render)
		terrain.setPos(-200, 0, -200)

		#self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")


	def spinCameraTask(self, task):
		#angleDegrees = task.time * -18.0
		#angleRadians = angleDegrees * (pi / 180.0)
		#self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
		#self.camera.setHpr(angleDegrees, 0, 0)
		print(self.camera.getHpr())
		print(self.camera.getPos())
		return Task.cont


world = World()
world.run()