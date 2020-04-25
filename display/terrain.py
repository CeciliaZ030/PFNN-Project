from direct.showbase.DirectObject import DirectObject
from panda3d.core import HeightfieldTesselator

def generate_terrain(filename="heightmaps/default.png"):
	tesselator = HeightfieldTesselator("Terrain")
	tesselator.setHeightfield(filename)
	tesselator.setHorizontalScale(1.0)
	tesselator.setVerticalScale(50.0)

	node = tesselator.generate()
	return node
