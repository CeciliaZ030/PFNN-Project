"""
ROOT hip
JOINT abdomen
	JOINT chest
		JOINT neck
			JOINT head
				JOINT leftEye
				JOINT rightEye
		JOINT rCollar
			JOINT rShldr
				JOINT rForeArm
					JOINT rHand
						JOINT rThumb1
							JOINT rThumb2
						JOINT rIndex1
							JOINT rIndex2
						JOINT rMid1
							JOINT rMid2
						JOINT rRing1
							JOINT rRing2
						JOINT rPinky1
							JOINT rPinky2
		JOINT lCollar
			JOINT lShldr
				JOINT lForeArm
					JOINT lHand
						JOINT lThumb1
							JOINT lThumb2
						JOINT lIndex1
							JOINT lIndex2
						JOINT lMid1	
							JOINT lMid2
						JOINT lRing1
							JOINT lRing2
						JOINT lPinky1
							JOINT lPinky2
JOINT rButtock
	JOINT rThigh
		JOINT rShin
			JOINT rFoot
JOINT lButtock
	JOINT lThigh
		JOINT lShin
			JOINT lFoot
"""
"""

"""
#JOINT_NUM = 43
JOINT_NUM = 21

#SDR_L, SDR_R, HIP_L, HIP_R = 22, 8, 39, 35
SDR_L, SDR_R, HIP_L, HIP_R = 10, 6, 17, 13
#FOOT_L = [42]
FOOT_L = [20, 20]
#FOOT_R = [38]
FOOT_R = [16, 16]
HEAD = 4
FILTER_OUT = ["leftEye", "rightEye", "rThumb1", "rThumb2", "rIndex1", "rIndex2", "rMid1", "rMid2", "rRing1", "rRing2", "rPinky1", "rPinky2",  "lThumb1", "lThumb2", "lIndex1", "lIndex2", "lMid1", "lMid2", "lRing1", "lRing2", "lPinky1", "lPinky2"]
JOINT_SCALE = 1

JOINT_WEIGHTS = [
	1, 
	1, 1, 1, 1, # 1e-10, 1e-10,
	1, 1e-10, 1, 1, # 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10,
	1, 1e-10, 1, 1, # 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10,
	1e-10, 1, 1, 1,
	1e-10, 1, 1, 1 ]
