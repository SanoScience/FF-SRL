import warp as wp
import math
import numpy as np
import FF_SRL as dk
from pxr import Usd, UsdGeom, Sdf, Vt, Gf

PI = wp.constant(math.pi)
ROTATIONCENTER_XFORM = wp.constant(0)
ROD_XFORM = wp.constant(3)
ACTION_ROD_FORWARD = wp.constant(0)
ACTION_ROD_ROTATE_CLOCKWISE = wp.constant(1)
ACTION_ROTATIONCENTER_X = wp.constant(2)
ACTION_ROTATIONCENTER_Y = wp.constant(3)
ACTION_CLAMP_OPEN = wp.constant(4)

CLAMP_DRAG_ANGLE = wp.constant(3)

@wp.kernel
def updateClampsDragKernel(positions: wp.array(dtype=wp.vec3),
                           laparoscopeInfo: wp.array(dtype = wp.vec3),
                           dragConstraints: wp.array(dtype = wp.float32),
                           lookupRadius: float,
                           stateConstraint: wp.array(dtype = wp.int32),
                           updateConstraint: wp.array(dtype = wp.int32)):

    if updateConstraint[0] == 0:
        return

    tid = wp.tid()

    if(stateConstraint[0] == 1):

        position = positions[tid]
        laparoscopeDragPoint = laparoscopeInfo[3]

        length = wp.length(position - laparoscopeDragPoint)

        if (length < lookupRadius):
            dragConstraints[tid] = 1.0

        else:
            dragConstraints[tid] = 0.0
    else:
        dragConstraints[tid] = 0.0

@wp.kernel
def transformLaparoscopeKernel(xform: wp.array(dtype=wp.mat44),
                               xformId: wp.int32,
                               actions: wp.array(dtype=wp.float32)):
    
    tXForm = wp.transpose(xform[xformId])

    # matrix = None

    if(xformId == ROTATIONCENTER_XFORM):
        matrix = getTransformationMatrix(0.0, 0.0, actions[0], 0.0, 0.0, actions[1])
        # matrix = getTransformationMatrix(0.0, 0.0, actions[ACTION_ROD_FORWARD], 0.0, 0.0, actions[ACTION_ROD_ROTATE_CLOCKWISE])
    elif(xformId == ROD_XFORM):
        matrix = getTransformationMatrix(0.0, 0.0, 0.0, actions[2], actions[3], 0.0)
        # matrix = getTransformationMatrix(0.0, 0.0, 0.0, actions[ACTION_ROTATIONCENTER_X], actions[ACTION_ROTATIONCENTER_Y], 0.0)

    xform[xformId] = wp.transpose(wp.mul(tXForm, matrix))

    # Apply transformation to children
    applyTransformationToChildren(xform, matrix, xformId)

@wp.kernel
def transformLaparoscopeGeneralKernel(xform: wp.array(dtype=wp.mat44),
                                      xformId: int,
                                      shiftX: float,
                                      shiftY: float,
                                      shiftZ: float,
                                      angleX: float,
                                      angleY: float,
                                      angleZ: float):
    
    tXForm = wp.transpose(xform[xformId])

    matrix = getTransformationMatrix(shiftX, shiftY, shiftZ, angleX, angleY, angleZ)

    xform[xformId] = wp.transpose(wp.mul(tXForm, matrix))

    # Apply transformation to children
    applyTransformationToChildren(xform, matrix, xformId)

@wp.kernel
def rotateClampsKernel(xform: wp.array(dtype=wp.mat44),
                       actions: wp.array(dtype=wp.float32),
                       maxAngle: wp.float32,
                       dragCutHeat: wp.array(dtype=wp.int32),
                       dragCutHeatUpdate: wp.array(dtype=wp.int32)):
    
    rodXForm = wp.transpose(xform[0])
    rodInvXForm = wp.inverse(rodXForm)

    leftClampXForm = wp.transpose(xform[4])
    rightClampXForm = wp.transpose(xform[5])
    rightClampLocalXForm = wp.mul(rodInvXForm, rightClampXForm)

    currentAngle = wp.asin(rightClampLocalXForm[2][1])*180.0/PI
    applyRotation = 0

    # diffAngle = actions[ACTION_CLAMP_OPEN]
    diffAngle = actions[4]

    # chceck if drag particles have been updated already
    if dragCutHeat[0] == 1:
        if dragCutHeatUpdate[0] == 1:
            dragCutHeatUpdate[0] = 0

    if diffAngle > 0.0:
        # Check if not exceeding max clamp angles
        if currentAngle < maxAngle:
            applyRotation = 1
        # Handle dragging
        if currentAngle > CLAMP_DRAG_ANGLE and dragCutHeat[0] == 1:
            dragCutHeat[0] = 0
            dragCutHeatUpdate[0] = 1
    else:
        # Check if not exceeding max clamp angles
        if currentAngle > 0:
            applyRotation = 1
        # Handle dragging
        if currentAngle < CLAMP_DRAG_ANGLE and dragCutHeat[0] == 0:
            dragCutHeat[0] = 1
            dragCutHeatUpdate[0] = 1

    if applyRotation > 0:
        matrix = getTransformationMatrix(0.0, 0.0, 0.0, diffAngle, 0.0, 0.0)
        matrixInv = getTransformationMatrix(0.0, 0.0, 0.0, -diffAngle, 0.0, 0.0)
        xform[4] = wp.transpose(wp.mul(leftClampXForm, matrixInv))
        applyTransformationToChildren(xform, matrixInv, 4)
        xform[5] = wp.transpose(wp.mul(rightClampXForm, matrix))
        applyTransformationToChildren(xform, matrix, 5)

@wp.kernel
def transformLaparoscopeMeshKernel(src: wp.array(dtype=wp.vec3),
                                   dest: wp.array(dtype=wp.vec3),
                                   xform: wp.array(dtype=wp.mat44),
                                   xFormId: int):

    tid = wp.tid()

    tXForm = wp.transpose(xform[xFormId])

    p = src[tid]
    m = wp.transform_point(tXForm, p)

    dest[tid] = m

@wp.kernel
def updateLaparoscopeKernel(xform: wp.array(dtype=wp.mat44),
                      laparoscopeTip: wp.array(dtype=wp.vec3),
                      laparoscopeBase: wp.array(dtype=wp.vec3),
                      laparoscopeRadius: wp.array(dtype=float),
                      laparoscopeHeights: wp.array(dtype=float)):

    tid = wp.tid()

    tXForm = wp.transpose(xform[tid])
    height = laparoscopeHeights[tid]

    if tid < 3:
        tXForm = wp.transpose(xform[tid])
        height = laparoscopeHeights[tid]
        baseLocal = wp.vec3(0.0, 0.0, 0.0 - height/ 2.0)
        tipLocal = wp.vec3(0.0, 0.0, 0.0 + height / 2.0)
    else:
        tXForm = wp.transpose(xform[0])
        rodHeight = laparoscopeHeights[0]
        clampHeight = laparoscopeHeights[2]
        baseLocal = wp.vec3(0.0, 0.0, 0.0 - rodHeight / 2.0 - 1.0 * clampHeight)
        tipLocal = wp.vec3(0.0, 0.0, 0.0 + rodHeight / 2.0 + 1.0 * clampHeight)

    laparoscopeBase[tid] = wp.transform_point(tXForm, baseLocal)
    laparoscopeTip[tid] = wp.transform_point(tXForm, tipLocal)

@wp.func
def getTransformationMatrix(shiftX: float, shiftY: float, shiftZ: float,
                            angleX: float, angleY: float, angleZ: float) -> wp.mat44:

    trans = wp.vec3(shiftX, shiftY, shiftZ)
    transZero = wp.vec3()
    rotZero = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 0.0), 0.0)
    rotX = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angleX)
    rotY = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angleY)
    rotZ = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angleZ)
    scale = wp.vec3(1.0, 1.0, 1.0)
    
    matrixShift = wp.mat44(trans, rotZero, scale)
    matrixX = wp.mat44(transZero, rotX, scale)
    matrixY = wp.mat44(transZero, rotY, scale)
    matrixZ = wp.mat44(transZero, rotZ, scale)

    return wp.mul(matrixX, wp.mul(matrixY, wp.mul(matrixZ, matrixShift)))

@wp.func
def applyTransformationToChildren(xform: wp.array(dtype=wp.mat44), matrix: wp.mat44, parentId: int) -> wp.int32:
    
    # todo: ugly but just for now
    if(parentId == 4):
            xform[1] = applyTransformationToChild(xform, matrix, parentId, 1)
    if(parentId == 5):
            xform[2] = applyTransformationToChild(xform, matrix, parentId, 2)
    if(parentId == 3):
        for i in range(6):
            if not i == parentId:
                xform[i] = applyTransformationToChild(xform, matrix, parentId, i)
    if(parentId == 0):
        for i in range(1, 6):
            if not i == 3:
                xform[i] = applyTransformationToChild(xform, matrix, parentId, i)

    return 1

@wp.func
def applyTransformationToChild(xform: wp.array(dtype=wp.mat44), matrix: wp.mat44, parentId: int, childId: int) -> wp.mat44:

    parentXForm = wp.transpose(xform[parentId])
    parentInvXForm = wp.inverse(parentXForm)
    childXForm = wp.transpose(xform[childId])

    # C'_gl = R'_gl * M * R'_gl^-1 * C_gl
    return wp.transpose(wp.mul(parentXForm, wp.mul(matrix, wp.mul(parentInvXForm, childXForm))))

class SimLaparoscopeDO(dk.SimObject):

    def __init__(self, prim: Usd.Prim, device: str = "cuda", stage=None, laparoscopeDragLookupRadius=0.15) -> None:
        super().__init__(prim, device)

        self.laparoscopeXForm = None
        self.laparoscopeInitialXForm = None
        self.laparoscopeBase = None
        self.laparoscopeInitialBase = None
        self.laparoscopeTip = None
        self.laparoscopeInitialTip = None
        self.laparoscopeRadius = None
        self.laparoscopeInitialRadius = None
        self.laparoscopeHeight = None
        self.laparoscopeInitialHeight = None
        self.laparoscopeDragCutHeat = None
        self.laparoscopeDragCutHeatUpdate = None
        self.laparoscopeDragLookupRadius = laparoscopeDragLookupRadius

        self.rodVertex = None
        self.rodVisPoint = None
        self.rodTriangle = None
        self.rodColor = None
        self.numRodVertices = None
        self.numRodTris = None

        self.laparoscopeScale = None

        self.hasLaparoscopeMesh = None

        self.rotationCenterPath = None
        self.rotationCenterPrim = None
        self.rotationCenterXForm = None

        self.rodPath = None
        self.rodPrim = None
        self.rodXForm = None

        self.rodCapsulePath = None
        self.rodCapsulePrim = None
        self.rodCapsuleXForm = None
        self.rodCapsuleBase = None
        self.rodCapsuleTip = None
        self.dragPoint = None

        self.leftClampRotationCenterPath = None
        self.leftClampRotationCenterPrim = None
        self.leftClampRotationCenterXForm = None
        self.leftClampPath = None
        self.leftClampPrim = None
        self.leftClampXForm = None
        self.leftClampBase = None
        self.leftClampTip = None
        self.leftClampColor = None

        self.rightClampRotationCenterPath = None
        self.rightClampRotationCenterPrim = None
        self.rightClampRotationCenterXForm = None
        self.rightClampPath = None
        self.rightClampPrim = None
        self.rightClampXForm = None
        self.rightClampBase = None
        self.rightClampTip = None
        self.rightClampColor = None

        self.texture = None

        time = Usd.TimeCode.Default()
        self.laparoscopeScale = prim.GetAttribute("xformOp:scale").Get()

        # Save world to local transformation
        self.initialXForm = self.xForm

        self.updateParameters(time, stage)
        self.transferDataToDevice()
        if prim.GetAttribute("hasLaparoscopeMesh").Get():
            self.hasLaparoscopeMesh = True
            self.readLaparoscopeMeshes(stage)
        else:
            self.hasLaparoscopeMesh = False
            self.generateCapsuleMeshes()

    def updateParameters(self, time, stage) -> None:
        # This is not yet available
        # for child in self.prim.GetAllDescendants():
        for child in stage.Traverse():
            if child.GetAttribute("simLaparoscope").Get() == True:
                laparoscopePrim = child
                self.rodColor = laparoscopePrim.GetAttribute("rodColor").Get()
                self.leftClampColor = laparoscopePrim.GetAttribute("leftClampColor").Get()
                self.rightClampColor = laparoscopePrim.GetAttribute("rightClampColor").Get()
            if child.GetAttribute("simLaparoscopeRotationCenter").Get() == True:
                self.rotationCenterPrim = child
                self.rotationCenterPath = self.rotationCenterPrim.GetPath()
                self.rotationCenterXForm = UsdGeom.Xformable(self.rotationCenterPrim).ComputeLocalToWorldTransform(time)
            if child.GetAttribute("simLaparoscopeRod").Get() == True:
                self.rodPrim = child
                self.rodPath = self.rodPrim.GetPath()
                self.rodXForm = UsdGeom.Xformable(self.rodPrim).ComputeLocalToWorldTransform(time)
            if child.GetAttribute("simLaparoscopeRodCapsule").Get() == True:
                self.rodCapsulePrim = child
                self.rodCapsulePath = self.rodCapsulePrim.GetPath()
                self.rodCapsuleXForm = UsdGeom.Xformable(self.rodCapsulePrim).ComputeLocalToWorldTransform(time)
                self.rodHeight = self.rodCapsulePrim.GetAttribute("height").Get()
                self.rodRadius = self.rodCapsulePrim.GetAttribute("radius").Get()

                rodBaseLocal = (0.0, 0.0, 0.0 - self.rodHeight / 2.0)
                rodTipLocal = (0.0, 0.0, 0.0 + self.rodHeight / 2.0)

                self.rodBase = self.rodXForm.Transform(Gf.Vec3f(rodBaseLocal))
                self.rodTip = self.rodXForm.Transform(Gf.Vec3f(rodTipLocal))
            if child.GetAttribute("simLaparoscopeLeftClampRotationCenter").Get() == True:
                self.leftClampRotationCenterPrim = child
                self.leftClampRotationCenterPath = self.leftClampRotationCenterPrim.GetPath()
                self.leftClampRotationCenterXForm = UsdGeom.Xformable(self.leftClampRotationCenterPrim).ComputeLocalToWorldTransform(time)
            if child.GetAttribute("simLaparoscopeLeftClamp").Get() == True:
                self.leftClampPrim = child
                self.leftClampPath = self.leftClampPrim.GetPath()
                self.leftClampXForm = UsdGeom.Xformable(self.leftClampPrim).ComputeLocalToWorldTransform(time)
                self.leftClampHeight = self.leftClampPrim.GetAttribute("height").Get()
                self.leftClampRadius = self.leftClampPrim.GetAttribute("radius").Get()

                # clampBaseLocal = (0.0, 0.0, 0.0 - self.leftClampHeight / 2.0)
                # clampTipLocal = (0.0, 0.0, 0.0 + self.leftClampHeight / 2.0)
                clampBaseLocal = (0.0, 0.0, 0.0 - self.leftClampHeight / 1.9)
                clampTipLocal = (0.0, 0.0, 0.0 + self.leftClampHeight / 1.9)
                self.leftClampBase = self.leftClampXForm.Transform(Gf.Vec3f(clampBaseLocal))
                self.leftClampTip = self.leftClampXForm.Transform(Gf.Vec3f(clampTipLocal))
            if child.GetAttribute("simLaparoscopeRightClampRotationCenter").Get() == True:
                self.rightClampRotationCenterPrim = child
                self.rightClampRotationCenterPath = self.rightClampRotationCenterPrim.GetPath()
                self.rightClampRotationCenterXForm = UsdGeom.Xformable(self.rightClampRotationCenterPrim).ComputeLocalToWorldTransform(time)
            if child.GetAttribute("simLaparoscopeRightClamp").Get() == True:
                self.rightClampPrim = child
                self.rightClampPath = self.rightClampPrim.GetPath()
                self.rightClampXForm = UsdGeom.Xformable(self.rightClampPrim).ComputeLocalToWorldTransform(time)
                self.rightClampHeight = self.rightClampPrim.GetAttribute("height").Get()
                self.rightClampRadius = self.rightClampPrim.GetAttribute("radius").Get()

                # clampBaseLocal = (0.0, 0.0, 0.0 - self.rightClampHeight / 2.0)
                # clampTipLocal = (0.0, 0.0, 0.0 + self.rightClampHeight / 2.0)
                clampBaseLocal = (0.0, 0.0, 0.0 - self.rightClampHeight / 1.9)
                clampTipLocal = (0.0, 0.0, 0.0 + self.rightClampHeight / 1.9)
                self.rightClampBase = self.rightClampXForm.Transform(Gf.Vec3f(clampBaseLocal))
                self.rightClampTip = self.rightClampXForm.Transform(Gf.Vec3f(clampTipLocal))

        dragPointLocal = (0.0, 0.0, 0.0 + self.rodHeight / 2.0 + 1.2 * self.rightClampHeight)
        self.dragPoint = self.rodXForm.Transform(Gf.Vec3f(dragPointLocal))

    def transferDataToDevice(self) -> None:
        laparoscopeXFormsList = [self.rodCapsuleXForm, self.leftClampXForm, self.rightClampXForm, self.rotationCenterXForm, self.leftClampRotationCenterXForm, self.rightClampRotationCenterXForm, self.rodXForm]
        laparoscopeBasesList = [self.rodBase, self.leftClampBase, self.rightClampBase, self.rodBase]
        laparoscopeTipsList = [self.rodTip, self.leftClampTip, self.rightClampTip, self.dragPoint]
        # laparoscopeRadiiList = [self.rodRadius, self.leftClampRadius, self.rightClampRadius, 0.0]
        laparoscopeRadiiList = [self.rodRadius * 2.0, self.leftClampRadius * 2.0, self.rightClampRadius * 2.0, 0.0]
        laparoscopeHeightsList = [self.rodHeight, self.leftClampHeight, self.rightClampHeight, 0.0]

        self.laparoscopeXForm = laparoscopeXFormsList
        self.laparoscopeBase = laparoscopeBasesList
        self.laparoscopeTip = laparoscopeTipsList
        self.laparoscopeRadius = laparoscopeRadiiList
        self.laparoscopeHeight = laparoscopeHeightsList

    def generateCapsuleMeshes(self) -> None:
                    
        laparoscopeRadiiList = [self.rodRadius, self.leftClampRadius, self.rightClampRadius]
        laparoscopeHeightsList = [self.rodHeight, self.leftClampHeight, self.rightClampHeight]

        capsuleVertices = []
        capsuleTriangles = []

        for i in range(3):
            capsuleVertex, capsuleTriangle = dk.generateTriMeshCapsule(radius=laparoscopeRadiiList[i], height=laparoscopeHeightsList[i], sectionsX=10, sectionsY=10)
            capsuleVertex = np.add(np.array(capsuleVertex), [0.0, 0.0, 0.0 - laparoscopeHeightsList[i]/2.0])

            capsuleVertices.append(capsuleVertex)
            capsuleTriangles.append(capsuleTriangle)

        # Allocate rod data
        self.numRodVisPoint = len(capsuleVertices[0])
        self.numRodVisFace = len(capsuleTriangles[0])

        # Allocate left clamp data
        self.numLeftClampVisPoint = len(capsuleVertices[1])
        self.numLeftClampVisFace = len(capsuleTriangles[1])

        # Allocate right clamp data
        self.numRightClampVisPoint = len(capsuleVertices[2])
        self.numRightClampVisFace = len(capsuleTriangles[2])

        capsuleTriangles[1] = [x + self.numRodVisPoint for x in capsuleTriangles[1]]
        capsuleTriangles[2] = [x + self.numRodVisPoint + self.numLeftClampVisPoint for x in capsuleTriangles[2]]

        self.visPoint = np.concatenate((capsuleVertices[0], capsuleVertices[1], capsuleVertices[2]))
        self.visFace = np.concatenate((capsuleTriangles[0], capsuleTriangles[1], capsuleTriangles[2]))

        self.numVisPoints = self.numRodVisPoint + self.numLeftClampVisPoint + self.numRightClampVisPoint
        self.numVisFaces = self.numRodVisFace + self.numLeftClampVisFace + self.numRightClampVisFace

    def readLaparoscopeMeshes(self, stage):

        linkVisPoint = None
        linkVisFace = None
        linkVisPointColor = None
        linkLocalXForm = None

        rodVisPoint = None
        rodVisFace = None
        rodVisPointColor = None
        rodLocalXForm = None

        leftClampVisPoint = None
        leftClampVisFace = None
        leftClampVisPointColor = None
        leftClampLocalXForm = None

        rightClampVisPoint = None
        rightClampVisFace = None
        rightClampVisPointColor = None
        rightClampLocalXForm = None

        for childPrim in stage.Traverse():
            if childPrim.GetTypeName() == "Mesh":
                if childPrim.GetAttribute("simLaparoscopeLinkMesh").Get():
                    linkVisPoint = list(childPrim.GetAttribute("points").Get())
                    linkVisFace = list(childPrim.GetAttribute("faceVertexIndices").Get())
                    linkVisPointColor = list(childPrim.GetAttribute("primvars:displayColor").Get())
                    linkLocalXForm = UsdGeom.Xformable(childPrim).GetLocalTransformation()
                if childPrim.GetAttribute("simLaparoscopeRodMesh").Get():
                    rodVisPoint = list(childPrim.GetAttribute("points").Get())
                    rodVisFace = list(childPrim.GetAttribute("faceVertexIndices").Get())
                    rodVisPointColor = list(childPrim.GetAttribute("primvars:displayColor").Get())
                    rodLocalXForm = UsdGeom.Xformable(childPrim).GetLocalTransformation()
                if childPrim.GetAttribute("simLaparoscopeLeftClampMesh").Get():
                    leftClampVisPoint = list(childPrim.GetAttribute("points").Get())
                    leftClampVisFace = list(childPrim.GetAttribute("faceVertexIndices").Get())
                    leftClampVisPointColor = list(childPrim.GetAttribute("primvars:displayColor").Get())
                    leftClampLocalXForm = UsdGeom.Xformable(childPrim).GetLocalTransformation()
                if childPrim.GetAttribute("simLaparoscopeRightClampMesh").Get():
                    rightClampVisPoint = list(childPrim.GetAttribute("points").Get())
                    rightClampVisFace = list(childPrim.GetAttribute("faceVertexIndices").Get())
                    rightClampVisPointColor = list(childPrim.GetAttribute("primvars:displayColor").Get())
                    rightClampLocalXForm = UsdGeom.Xformable(childPrim).GetLocalTransformation()

        # Tranform meshes
        linkVisPoint = dk.transformVecArray(linkVisPoint, linkLocalXForm)
        rodVisPoint = dk.transformVecArray(rodVisPoint, rodLocalXForm)
        leftClampVisPoint = dk.transformVecArray(leftClampVisPoint, leftClampLocalXForm)
        rightClampVisPoint = dk.transformVecArray(rightClampVisPoint, rightClampLocalXForm)

        # Allocate rod data
        self.numRodVisPoint = len(linkVisPoint + rodVisPoint)
        self.numRodVisFace = int(len(linkVisFace + rodVisFace)/3)

        # Allocate left clamp data
        self.numLeftClampVisPoint = len(leftClampVisPoint)
        self.numLeftClampVisFace = int(len(leftClampVisFace)/3)

        # Allocate right clamp data
        self.numRightClampVisPoint = len(rightClampVisPoint)
        self.numRightClampVisFace = int(len(rightClampVisFace)/3)

        self.visPoint = linkVisPoint + rodVisPoint + leftClampVisPoint + rightClampVisPoint
        rodVisFace = [x + len(linkVisPoint) for x in rodVisFace]
        leftClampVisFace = [x + self.numRodVisPoint for x in leftClampVisFace]
        rightClampVisFace = [x + self.numRodVisPoint + self.numLeftClampVisPoint for x in rightClampVisFace]

        self.visFace = linkVisFace + rodVisFace + leftClampVisFace + rightClampVisFace
        self.visPointColor = linkVisPointColor + rodVisPointColor + leftClampVisPointColor + rightClampVisPointColor

        self.numVisPoints = self.numRodVisPoint + self.numLeftClampVisPoint + self.numRightClampVisPoint
        self.numVisFaces = self.numRodVisFace + self.numLeftClampVisFace + self.numRightClampVisFace