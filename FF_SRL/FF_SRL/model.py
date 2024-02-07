import warp as wp
import math
import numpy as np
import FF_SRL as dk
import torch
from pxr import Usd, UsdGeom, Sdf, Vt, Gf, UsdSkel

in_de_crease_steps = 5

class SimObject:
    """Describes a general simulation object
    """

    def __init__(self, prim:Usd.Prim, device:str="cuda") -> None:
        """_summary_

        Args:
            xForm (_type_): _description_
        """

        # General, meta data
        self.prim = prim
        self.path = self.prim.GetPath()
        self.name = self.prim.GetAttribute("name").Get()
        self.device = device

        # xform properties
        self.xForm = None
        self.xFormApi = None
        self.translation = None
        self.rotation = None
        self.scale = None
        self.pivot = None
        self.rotOrder = None

        time = Usd.TimeCode.Default()
        self.updateXForm(time)
        # self.initialized()

    def updateXForm(self, time) -> None:
        self.xForm = UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(time)
        self.xFormAPI = UsdGeom.XformCommonAPI(self.prim)

        self.translation, self.rotation, self.scale = UsdSkel.DecomposeTransform(self.xForm)
        self.rotation = Gf.Rotation(self.rotation).Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))

    def transformLocalVecArray(self, points):
        return dk.transformVecArray(points, self.xForm)

    def transformGlobalVecArray(self, points):
        return dk.transformVecArray(points, self.xForm.GetInverse())

    def transformLocalVecArrayWarp(self, srcPoints, outPoints) -> None:
        dk.launchTransformVecArrayWarp(srcPoints, outPoints, self.xForm, self.device)

    def transformGlobalVecArrayWarp(self, srcPoints, outPoints) -> None:
        dk.launchTransformVecArrayWarp(srcPoints, outPoints, self.xForm.GetInverse(), self.device)

    def initialized(self) -> None:
        print("Initialized object: " + self.name + " at position: " + str(self.translation))

class SimLockBox(SimObject):

    def __init__(self, prim:Usd.Prim, device:str="cuda") -> None:
        super().__init__(prim, device)

    def lockArray(self, vecArray, array):

        # p1, p2, p3, p4 = dk.getBoxSpanVectors(self.xForm, self.scale[0])
        p1, p2, p3, p4 = dk.getBoxSpanVectors(self.xForm)

        pi = p2 - p1
        pj = p3 - p1
        pk = p4 - p1

        for i in range(len(vecArray)):
            vec = Gf.Vec3f(vecArray[i].tolist())
            v = vec - p1

            if dk.checkIfWithinBounds(v, pi, pj, pk):
                array[i] = 0.0

class SimMesh(SimObject):

    def __init__(self, prim:Usd.Prim, device:str="cuda:0", requiresGrad=True) -> None:
        super().__init__(prim, device)

        ## Global
        self.translationWP = None
        self.rotationWP = None
        self.requiresGrad = requiresGrad

        ## Mesh parameters
        self.density = None
        self.ksDistance = None
        self.ksVolume = None
        self.ksContact = None
        self.ksDrag = None
        self.volumeCompliance = None
        self.deviatoricCompliance = None
        self.breakFactor = None
        self.breakThreshold = None
        self.heatLoss = None
        self.heatConduction = None
        self.heatTransfer = None
        self.heatLimit = None
        self.heatSphereRadius = None
        self.heatQuantum = None

        ## Mesh mapping
        self.pointToVertex = None
        self.pointToVertexHost = None
        self.vertexToPoint = None
        self.vertexToPointHost = None
        self.faceToTriangle = None
        self.faceToTriangleHost = None
        self.triangleToFace = None
        self.triangleToFaceHost = None
        self.edgesToTriangles = None
        self.edgesToTrianglesHost = None
        self.tetrahedronNeighbors = None
        self.tetrahedronNeighborsHost = None
        self.triangleToTetrahedron = None
        self.triangleToTetrahedronHost = None
        self.tetrahedronToTriangle = None
        self.tetrahedronToTriangleHost = None

        ## Mesh data
        self.meshVisPoints = None
        self.meshNormals = None
        self.meshVisFaces = None

        # Points
        self.vertex = None
        self.initialVertexHost = None
        self.numVertices = None
        self.visPoint = None
        self.initialVisPoint = None
        self.numVisPoints = None
        self.normal = None
        self.dP = None
        self.predictedVertex = None
        self.outputVertex = None
        self.velocity = None
        self.inverseMass = None
        self.inverseMassHost = None

        # Edges
        self.numEdges = None
        self.edge = None
        self.edgeLambda = None
        self.edgeRestLength = None
        self.activeEdge = None
        self.flipEdge = None

        # Triangles/Faces
        self.numTris = None
        self.triangle = None
        self.hole = None

        self.numVisFaces = None
        self.visFace = None
        self.visHole = None

        # Tetrahedrons
        self.numTetrahedrons = None
        self.tetrahedron = None
        self.tetrahedronLambda = None
        self.tetrahedronRestVolume = None
        self.tetrahedronInverseMassNeoHookean = None
        self.tetrahedronInverseMassNeoHookeanHost = None
        self.tetrahedronInverseRestPositionNeoHookean = None
        self.neoHookeanLambda = None
        self.activeTetrahedron = None
        self.flipTetrahedron = None

        # Constraints
        self.constraintsNumber = None
        self.activeDragConstraint = None
        self.flipDragConstraint = None

        # Rendering
        self.texCoords = None
        self.texIndices = None

        self.getMeshProperties()
        self.getMeshMapping()
        self.getMeshData()

    def getMeshProperties(self):
        self.density = self.prim.GetAttribute("param:density").Get()
        self.ksDistance = self.prim.GetAttribute("param:ksDistance").Get()
        self.ksVolume = self.prim.GetAttribute("param:ksVolume").Get()
        self.ksContact = self.prim.GetAttribute("param:ksContact").Get()
        self.ksDrag = self.prim.GetAttribute("param:ksDrag").Get()
        self.volumeCompliance = self.prim.GetAttribute("param:volumeCompliance").Get()
        self.deviatoricCompliance = self.prim.GetAttribute("param:deviatoricCompliance").Get()
        self.breakFactor = self.prim.GetAttribute("param:breakFactor").Get()
        self.breakThreshold = self.prim.GetAttribute("param:breakThreshold").Get()
        self.heatLoss = self.prim.GetAttribute("param:heatLoss").Get()
        self.heatConduction = self.prim.GetAttribute("param:heatConduction").Get()
        self.heatTransfer = self.prim.GetAttribute("param:heatTransfer").Get()
        self.heatLimit = self.prim.GetAttribute("param:heatLimit").Get()  
        self.heatSphereRadius = self.prim.GetAttribute("param:heatSphereRadius").Get()  
        self.heatQuantum = self.prim.GetAttribute("param:heatQuantum").Get()

    def getMeshMapping(self):
        meshPointToVertex = self.prim.GetAttribute("mapping:pointToVertex").Get()
        meshEdgesToTriangles = self.prim.GetAttribute("mapping:edgeToTriangles").Get()
        meshVertexToPoint = self.prim.GetAttribute("mapping:vertexToPoint").Get()
        meshTriangleToFace = self.prim.GetAttribute("mapping:triangleToFace").Get()
        meshFaceToTriangle = self.prim.GetAttribute("mapping:faceToTriangle").Get()
        meshTriangleToTetrahedron = self.prim.GetAttribute("mapping:triangleToTetrahedron").Get()
        meshTetrahedronToTriangle = self.prim.GetAttribute("mapping:tetrahedronToTriangles").Get()
        meshTetrahedronNeighbors = self.prim.GetAttribute("mapping:tetrahedronNeighbors").Get()

        meshPointToVertex = np.array(meshPointToVertex)
        meshPointToVertex = meshPointToVertex.ravel()
        self.pointToVertex = wp.array(meshPointToVertex, dtype=wp.int32, device=self.device)
        self.pointToVertexHost = wp.array(meshPointToVertex, dtype=wp.int32, device='cpu')

        meshVertexToPoint = np.array(meshVertexToPoint)
        meshVertexToPoint = meshVertexToPoint.ravel()
        self.vertexToPoint = wp.array(meshVertexToPoint, dtype=wp.int32, device=self.device)
        self.vertexToPointHost = wp.array(meshVertexToPoint, dtype=wp.int32, device='cpu')

        meshFaceToTriangle = np.array(meshFaceToTriangle)
        meshFaceToTriangle = meshFaceToTriangle.ravel()
        self.faceToTriangle = wp.array(meshFaceToTriangle, dtype=wp.int32, device=self.device)
        self.faceToTriangleHost = wp.array(meshFaceToTriangle, dtype=wp.int32, device='cpu')

        meshTriangleToFace = np.array(meshTriangleToFace)
        meshTriangleToFace = meshTriangleToFace.ravel()
        self.triangleToFace = wp.array(meshTriangleToFace, dtype=wp.int32, device=self.device)
        self.triangleToFaceHost = wp.array(meshTriangleToFace, dtype=wp.int32, device='cpu')

        meshEdgesToTriangles = np.array(meshEdgesToTriangles)
        meshEdgesToTriangles = meshEdgesToTriangles.ravel()
        self.edgesToTriangles = wp.array(meshEdgesToTriangles, dtype=wp.int32, device=self.device)
        self.edgesToTrianglesHost = wp.array(meshEdgesToTriangles, dtype=wp.int32, device='cpu')

        meshTriangleToTetrahedron = np.array(meshTriangleToTetrahedron)
        meshTriangleToTetrahedron = meshTriangleToTetrahedron.ravel()
        self.triangleToTetrahedron = wp.array(meshTriangleToTetrahedron, dtype=wp.int32, device=self.device)
        self.triangleToTetrahedronHost = wp.array(meshTriangleToTetrahedron, dtype=wp.int32, device='cpu')

        meshTetrahedronToTriangle = np.array(meshTetrahedronToTriangle)
        meshTetrahedronToTriangle = meshTetrahedronToTriangle.ravel()
        self.tetrahedronToTriangle = wp.array(meshTetrahedronToTriangle, dtype=wp.int32, device=self.device)
        self.tetrahedronToTriangleHost = wp.array(meshTetrahedronToTriangle, dtype=wp.int32, device='cpu')

        meshTetrahedronNeighbors = np.array(meshTetrahedronNeighbors)
        meshTetrahedronNeighbors = meshTetrahedronNeighbors.ravel()
        self.tetrahedronNeighbors = wp.array(meshTetrahedronNeighbors, dtype=wp.int32, device=self.device)
        self.tetrahedronNeighborsHost = wp.array(meshTetrahedronNeighbors, dtype=wp.int32, device='cpu')

    def getMeshData(self):
        self.meshVisPoints = self.prim.GetAttribute("points").Get()
        self.meshNormals = self.prim.GetAttribute("normals").Get()
        meshVisFaces = self.prim.GetAttribute("faceVertexIndices").Get()
        meshVisFaces = np.array(meshVisFaces)
        self.meshVisFaces = meshVisFaces.ravel()
        meshVertices = self.prim.GetAttribute("extMesh:vertex").Get()
        meshTriangles = self.prim.GetAttribute("extMesh:triangle").Get()
        meshInverseMasses = self.prim.GetAttribute("extMesh:inverseMass").Get()
        meshEdges = self.prim.GetAttribute("extMesh:edge").Get()
        meshEdges = np.array(meshEdges)
        meshEdges = meshEdges.ravel()
        meshEdgesRestLengths = self.prim.GetAttribute("extMesh:edgeRestLength").Get()
        meshTetrahedrons = self.prim.GetAttribute("extMesh:elem").Get()
        meshTetrahedrons = np.array(meshTetrahedrons)
        meshTetrahedrons = meshTetrahedrons.ravel()
        meshTetrahedronsRestVolumes = self.prim.GetAttribute("extMesh:tetrahedronRestVolume").Get()
        meshTetrahedronsInverseMassesNeoHookean = self.prim.GetAttribute("extMesh:inverseMassNeoHookean").Get()
        meshTetrahedronsInverseRestPositionsNeoHookean = self.prim.GetAttribute("extMesh:inverseRestPosition").Get()
        meshTetrahedronsInverseRestPositionsNeoHookean = np.array(meshTetrahedronsInverseRestPositionsNeoHookean).reshape((-1, 3, 3))

        self.numVertices = len(meshVertices)
        self.numVisPoints = len(self.meshVisPoints)
        self.numTris = int(len(meshTriangles) / 3)
        self.numVisFaces = int(len(meshVisFaces) / 3)
        self.numEdges = int(len(meshEdges) / 2)
        self.numTetrahedrons = int(len(meshTetrahedrons) / 4)

        meshDragConstraints = np.zeros(self.numVertices)
        meshActiveTetrahedrons = np.ones(self.numTetrahedrons)
        meshActiveEdges = np.ones(self.numEdges)
        meshVisColors = np.ones(self.numVisPoints*3, dtype=float)

        # transform particles to world space
        meshVerticesWorld = self.transformLocalVecArray(meshVertices)

        # convert node inputs to GPU arrays
        self.vertex = wp.array(meshVerticesWorld, dtype=wp.vec3, device=self.device, requires_grad=self.requiresGrad)
        self.initialVertex = wp.array(meshVerticesWorld, dtype=wp.vec3, device=self.device, requires_grad=self.requiresGrad)
        self.initialVertexHost = wp.array(meshVerticesWorld, dtype=wp.vec3, device="cpu")
        self.normal = wp.array(self.meshNormals, dtype=wp.vec3, device=self.device)
        self.initialVisPoint = wp.array(self.meshVisPoints, dtype=wp.vec3, device=self.device)
        self.visPoint = wp.array(self.meshVisPoints, dtype=wp.vec3, device=self.device)
        self.outputVertex = wp.array(meshVerticesWorld, dtype=wp.vec3, device=self.device)
        self.predictedVertex = wp.zeros_like(self.vertex, requires_grad=self.requiresGrad)
        self.velocity = wp.zeros_like(self.vertex, requires_grad=self.requiresGrad)
        self.triangle = wp.array(meshTriangles, dtype=int, device=self.device)
        self.visFace = wp.array(meshVisFaces, dtype=int, device=self.device)
        self.visColor = wp.array(meshVisColors, dtype=float, device=self.device)
        self.inverseMass = wp.array(meshInverseMasses, dtype=float, device=self.device)
        self.inverseMassHost = wp.array(meshInverseMasses, dtype=float, device="cpu")
        self.dP = wp.zeros_like(self.vertex)
        self.constraintsNumber = wp.zeros(self.numVertices, dtype=int, device=self.device)
        self.hole = wp.zeros(self.numTris, dtype=wp.int32, device=self.device)

        self.translationWP = wp.array(wp.vec3(self.translation[0], self.translation[1], self.translation[2]), dtype=wp.vec3, device=self.device)
        self.rotationWP = wp.array(wp.quat_rpy(self.rotation[2], self.rotation[0], self.rotation[1]), dtype=wp.quat, device=self.device)

        # for renderer tests
        distance = np.linalg.norm(self.meshVisPoints, axis=1)
        radius = np.max(distance)
        distance = distance / radius
        tex_coords = np.stack((distance, distance), axis=1)
        tex_indices = meshVisFaces

        self.texCoords = wp.array(tex_coords, dtype=wp.vec2, device=self.device)
        self.texIndices = wp.array(tex_indices, dtype=int, device=self.device)

        self.activeDragConstraint = wp.array(meshDragConstraints, dtype=float, device=self.device)
        self.flipDragConstraint = wp.zeros(self.numVertices, dtype=int, device=self.device)

        self.edge = wp.array(meshEdges, dtype=int, device=self.device)
        self.edgeRestLength = wp.array(meshEdgesRestLengths, dtype=float, device=self.device)
        self.edgeLambda = wp.zeros(self.numEdges, dtype=float, device=self.device)
        self.activeEdge = wp.array(meshActiveEdges, dtype=float, device=self.device)
        self.flipEdge = wp.zeros(self.numEdges, dtype=int, device=self.device)

        self.tetrahedron = wp.array(meshTetrahedrons, dtype=int, device=self.device)
        self.tetrahedronRestVolume = wp.array(meshTetrahedronsRestVolumes, dtype=float, device=self.device)
        self.tetrahedronLambda = wp.zeros(self.numTetrahedrons, dtype=float, device=self.device)
        self.neoHookeanLambda = wp.zeros(self.numTetrahedrons, dtype=float, device=self.device)
        self.tetrahedronInverseMassNeoHookean = wp.array(meshTetrahedronsInverseMassesNeoHookean, dtype=float, device=self.device)
        self.tetrahedronInverseMassNeoHookeanHost = wp.array(meshTetrahedronsInverseMassesNeoHookean, dtype=float, device="cpu")
        self.tetrahedronInverseRestPositionNeoHookean = wp.array(meshTetrahedronsInverseRestPositionsNeoHookean, dtype=wp.mat33, device=self.device)
        self.activeTetrahedron = wp.array(meshActiveTetrahedrons, dtype=float, device=self.device)
        self.flipTetrahedron = wp.zeros(self.numTetrahedrons, dtype=int, device=self.device)

        self.edgeBreaker = wp.zeros(self.numEdges, dtype=float, device=self.device)
        self.tetrahedronBreaker = wp.zeros(self.numTetrahedrons, dtype=float, device=self.device)
        self.vertexHeat = wp.zeros(self.numVertices, dtype=float, device=self.device)
        self.tetrahedronHeat = wp.zeros(self.numTetrahedrons, dtype=float, device=self.device)
        self.tetrahedronNewHeat = wp.zeros(self.numTetrahedrons, dtype=float, device=self.device)

        # self.debugContactVector = wp.zeros(self.numVertices, dtype=wp.vec3, device=self.device)
        # self.debugContactTriangle = wp.zeros(self.numTris, dtype=wp.int32, device=self.device)

    def updateInverseMass(self, newMass) -> None:
        self.inverseMass = wp.array(newMass, dtype=float, device=self.device)

    def updateInverseMassNeoHookean(self, newMass) -> None:
        self.tetrahedronInverseMassNeoHookean = wp.array(newMass, dtype=float, device=self.device)

    def remapPointsFromVerticesWarp(self) -> None:
        dk.launchRemapAToB(self.visPoint, self.outputVertex, self.pointToVertex, self.device)

    def remapGlobalPointsFromVerticesWarp(self) -> None:
        dk.launchRemapAToB(self.visPoint, self.vertex, self.pointToVertex, self.device)

    def transformMeshData(self) -> None:
        self.transformGlobalVecArrayWarp(self.vertex, self.outputVertex)
        self.remapPointsFromVerticesWarp()

    def resetFixed(self) -> None:
        zero3Vector = wp.zeros_like(self.vertex)
        zeroFloat = wp.zeros_like(self.inverseMass)

        wp.copy(self.vertex, self.initialVertex)
        wp.copy(self.visPoint, self.initialVisPoint)
        # wp.copy(self.velocity, zero3Vector)
        wp.copy(self.predictedVertex, zero3Vector)
        wp.copy(self.dP, zeroFloat)
        wp.copy(self.activeDragConstraint, zeroFloat)
        wp.copy(self.flipDragConstraint, zeroFloat)

class SimConnector(SimObject):
    def __init__(self, prim: Usd.Prim, device: str = "cuda") -> None:
        super().__init__(prim, device)

        self.pointBodyPath = None
        self.triBodyPath = None

        self.massWeightRatio = None
        self.kS = None
        self.restLengthMul = None

        self.numTris = None

        self.pointIds = None
        self.triIds = None
        self.triBar = None
        self.restDist = None

        self.pointBodyPredictedVertex = None
        self.pointBodyDP = None
        self.pointBodyConstraintsNumber = None
        self.triBodyPredictedVertex = None
        self.triBodyDP = None
        self.triBodyConstraintsNumber = None

        self.updateConnector()

    def updateConnector(self) -> None:

        self.pointBodyPath = self.prim.GetRelationship("pointBody").GetTargets()[0]
        self.triBodyPath = self.prim.GetRelationship("triBody").GetTargets()[0]

        self.massWeightRatio = self.prim.GetAttribute("param:massWeightRatio").Get()
        self.kS = self.prim.GetAttribute("param:kS").Get()
        self.restLengthMul = self.prim.GetAttribute("param:restLengthMul").Get()

        pointIds = self.prim.GetAttribute("pointId").Get()
        triIds = np.array(self.prim.GetAttribute("triIds").Get()).ravel()
        triBar = self.prim.GetAttribute("triBar").Get()
        restDist = self.prim.GetAttribute("restDist").Get()

        self.numTris = int(len(triIds)/3)

        self.pointIds = wp.array(pointIds, dtype=int, device=self.device)
        self.triIds = wp.array(triIds, dtype=int, device=self.device)
        self.triBar = wp.array(triBar, dtype=wp.vec3, device=self.device)
        self.restDist = wp.array(restDist, dtype=float, device=self.device)

    def updatePointInfo(self, simMesh: SimMesh) -> None:
        self.pointBodyPredictedVertex = simMesh.predictedVertex
        self.pointBodyDP = simMesh.dP
        self.pointBodyConstraintsNumber = simMesh.constraintsNumber

    def updateTriInfo(self, simMesh: SimMesh) -> None:
        self.triBodyPredictedVertex = simMesh.predictedVertex
        self.triBodyDP = simMesh.dP
        self.triBodyConstraintsNumber = simMesh.constraintsNumber

class SimElasticRod(SimObject):
    def __init__(self, prim: Usd.Prim, device: str = "cuda") -> None:
        super().__init__(prim, device)

        ## Parameters
        self.element = None
        self.elementNum = None
        self.elementRadius = None
        self.elementHeight = None
        self.elementKs = None
        self.iterativeRod = None

        ## Data
        self.elementVertex = None
        self.elementInitialVertex = None
        self.elementPredictedVertex = None
        self.elementOutputVertex = None
        self.elementDp = None
        self.elementEdge = None
        self.elementEdgeRestLength = None
        self.elementActiveEdge = None
        self.elementConstraintsNumber = None
        self.elementEdgeLambda = None
        self.elementVelocity = None
        self.elementInverseMass = None
        self.elementInverseMassHost = None
        self.numElementEdge = None

        time = Usd.TimeCode.Default()
        self.element = self.prim.GetChildren()
        self.updateParameters()
        self.updateData(time)

    def updateData(self, time) -> None:
        elementInverseMass = self.prim.GetAttribute("elementInverseMass").Get()
        elementVertex = []
        elementInitialVertex = []
        elementInvMass = []

        for element in self.element:
            elementXform = UsdGeom.Xformable(element).ComputeLocalToWorldTransform(time)
            elementPos = [elementXform[3][0], elementXform[3][1], elementXform[3][2]]
            elementVertex.append(elementPos)
            elementInitialVertex.append([elementPos[0] - self.translation[0],
                                         elementPos[1] - self.translation[1],
                                         elementPos[2] - self.translation[2]])
            elementInvMass.append(elementInverseMass)
        elementInvMass[0] = 0.0

        elementEdge = []
        elementEdgeRestLength = []
        elementActiveEdge = []
        for i in range(len(self.element) - 1):
            elementEdge.append(i)
            elementEdge.append(i + 1)
            elementEdgeRestLength.append(self.elementHeight)
            elementActiveEdge.append(1.0)

        self.elementVertex = wp.array(elementVertex, dtype=wp.vec3, device=self.device)
        self.elementInitialVertex = wp.array(elementInitialVertex, dtype=wp.vec3, device=self.device)
        self.elementPredictedVertex = wp.zeros_like(self.elementVertex)
        self.elementOutputVertex = wp.zeros_like(self.elementVertex)
        self.elementDp = wp.zeros_like(self.elementVertex)
        self.elementEdge = wp.array(elementEdge, dtype=int, device=self.device)
        self.elementEdgeRestLength = wp.array(elementEdgeRestLength, dtype=float, device=self.device)
        self.elementActiveEdge = wp.array(elementActiveEdge, dtype=float, device=self.device)
        self.elementConstraintsNumber = wp.zeros(len(elementEdge), dtype=int, device=self.device)
        self.elementEdgeLambda = wp.zeros(len(elementEdge), dtype=float, device=self.device)
        self.elementVelocity = wp.zeros_like(self.elementVertex)
        self.elementInverseMass = wp.array(elementInvMass, dtype=float, device=self.device)
        self.elementInverseMassHost = wp.array(elementInvMass, dtype=float, device="cpu")

        self.numElementEdge = int(len(self.elementEdge) / 2.0)

    def updateParameters(self) -> None:
        self.elementNum = len(self.element)
        self.elementRadius = self.prim.GetAttribute("rodRadius").Get()
        self.elementHeight = self.prim.GetAttribute("elementLength").Get()
        self.elementKs = self.prim.GetAttribute("rodKs").Get()
        self.iterativeRod = self.prim.GetAttribute("iterativeRod").Get()

    def updateInverseMass(self, newMasses) -> None:
        self.elementInverseMass = wp.array(newMasses, dtype=float, device=self.device)

class SimContext():
    def __init__(self) -> None:
        self.simSubsteps = None
        self.simFrameRate = None
        self.simConstraints = None
        self.simDt = None
        self.simDevice = None

        self.gravity = None
        self.velocityDampening = None
        self.groundLevel = None
        self.globalKsDistance = None
        self.globalDistanceCompliance = None
        self.globalKsVolume = None
        self.globalVolumeCompliance = None
        self.globalKsContact = None
        self.globalKsDrag = None
        self.globalVolumeComplianceNeoHookean = None
        self.gloabalDeviatoricComplianceNeoHookean = None

        self.globalBreakFactor = None
        self.globalBreakThreshold = None
        self.globalHeatLoss = None
        self.globalHeatConduction = None
        self.globalHeatTransfer = None
        self.globalHeatLimit = None
        self.globalHeatSphereRadius = None
        self.globalHeatQuantum = None

    def readFromDBInputs(self, dbInputs) -> None:
        self.simSubsteps = dbInputs.simSubsteps
        self.simFrameRate = dbInputs.simFrameRate
        self.simConstraints = dbInputs.simConstraintsSteps
        self.simDt = (1.0/self.simFrameRate)/self.simSubsteps

        self.gravity = dbInputs.environmentGravity
        self.velocityDampening = dbInputs.environmentVelocityDampening
        self.groundLevel = dbInputs.environmentGroundLevel
        self.globalKsDistance = dbInputs.globalKsDistance
        self.globalDistanceCompliance = dbInputs.globalKsDistance / (self.simDt * self.simDt)
        self.globalKsVolume = dbInputs.globalKsVolume
        self.globalVolumeCompliance = dbInputs.globalKsVolume / (self.simDt * self.simDt)
        self.globalKsContact = dbInputs.globalKsContact
        self.globalKsDrag = dbInputs.globalKsDrag
        self.globalVolumeComplianceNeoHookean = dbInputs.globalVolumeCompliance
        self.gloabalDeviatoricComplianceNeoHookean = dbInputs.globalDeviatoricCompliance

        self.globalBreakFactor = dbInputs.globalBreakFactor
        self.globalBreakThreshold = dbInputs.globalBreakThreshold
        self.globalHeatLoss = dbInputs.globalHeatLoss
        self.globalHeatConduction = dbInputs.globalHeatConduction
        self.globalHeatTransfer = dbInputs.globalHeatTransfer
        self.globalHeatLimit = dbInputs.globalHeatLimit
        self.globalHeatSphereRadius = dbInputs.globalHeatSphereRadius
        self.globalHeatQuantum = dbInputs.globalHeatQuantum

    def setSimDevice(self, device: str) -> None:
        self.simDevice = device

class SimInteractions():
    def __init__(self) -> None:
        self.isDragging = False
        self.isCutting = False
        self.isHeating = False

class SimModel():
    def __init__(self, stage, numEnvs, device, 
                 simSubsteps=8,
                 simFrameRate=60,
                 simConstraintsSteps=1,
                 environmentGravity=(0.0, -9.8, 0.0),
                 environmentVelocityDampening=0.1,
                 environmentGroundLevel=0.0,
                 globalKsDistance=1.0,
                 globalKsVolume=1.0,
                 globalKsContact=1.0,
                 globalKsDrag=1.0,
                 globalVolumeCompliance=1.0,
                 globalDeviatoricCompliance=1.0,
                 globalBreakFactor=1.0,
                 globalBreakThreshold=1.0,
                 globalHeatLoss=1.0,
                 globalHeatConduction=1.0,
                 globalHeatTransfer=1.0,
                 globalHeatLimit=1.0,
                 globalHeatSphereRadius=1.0,
                 globalHeatQuantum=1.0,
                 globalLaparoscopeDragLookupRadius=0.15) -> None:

        self.simEnvironments = []
        self.stage = stage
        self.device = device
        self.numEnvs = numEnvs

        self.simSubsteps = simSubsteps
        self.simFrameRate = simFrameRate
        self.simConstraints = simConstraintsSteps
        self.simDt = (1.0/self.simFrameRate)/self.simSubsteps

        self.gravity = environmentGravity
        self.velocityDampening = environmentVelocityDampening
        self.groundLevel = environmentGroundLevel
        self.globalKsDistance = globalKsDistance
        self.globalDistanceCompliance = globalKsDistance / (self.simDt * self.simDt)
        self.globalKsVolume = globalKsVolume
        self.globalVolumeCompliance = globalKsVolume / (self.simDt * self.simDt)
        self.globalKsContact = globalKsContact
        self.globalKsDrag = globalKsDrag
        self.globalVolumeComplianceNeoHookean = globalVolumeCompliance
        self.gloabalDeviatoricComplianceNeoHookean = globalDeviatoricCompliance

        self.globalLaparoscopeDragLookupRadius = globalLaparoscopeDragLookupRadius

        self.globalBreakFactor = globalBreakFactor
        self.globalBreakThreshold = globalBreakThreshold
        self.globalHeatLoss = globalHeatLoss
        self.globalHeatConduction = globalHeatConduction
        self.globalHeatTransfer = globalHeatTransfer
        self.globalHeatLimit = globalHeatLimit
        self.globalHeatSphereRadius = globalHeatSphereRadius
        self.globalHeatQuantum = globalHeatQuantum

        for i in range(numEnvs):
            simEnvironment = dk.SimEnvironment(self.stage, self.device, i, globalLaparoscopeDragLookupRadius=self.globalLaparoscopeDragLookupRadius)
            self.simEnvironments.append(simEnvironment)

    def duplicateVisibleObjects(self, spacingX:float=1.0, spacingZ:float=1.0):
        for simEnvironment in self.simEnvironments:
            simEnvironment.removeOriginalMeshes()
            # simEnvironment.removeOriginalLaparoscopes()
            simEnvironment.duplicateMeshes(spacingX, spacingZ)
            simEnvironment.duplicateLaparoscopes(spacingX, spacingZ)

    def applyActions(self, actions):
        for simEnvironment in self.simEnvironments:
            simEnvironment.applyActions(actions)

    # def getModelObservationsTensor(self):
    #     modelObservationsList = []
    #     for simEnvironment in self.simEnvironments:
    #         modelObservationsList.append(simEnvironment.getEnvironmentObservationsTensor())

    #     modelObservationsTensor = torch.cat(modelObservationsList)
    #     return modelObservationsTensor

    def getModelObservationsTensor(self):
        modelObservationsList = []
        for simEnvironment in self.simEnvironments:
            modelObservationsList.append(simEnvironment.getLaparoscopeObservationsTensor())

        modelObservationsTensor = torch.cat(modelObservationsList)
        return modelObservationsTensor

    def getModelRewardTensor(self):
        modelRewardList = []
        for simEnvironment in self.simEnvironments:
            modelRewardList.append(simEnvironment.getEnvironmentRewardTensor())

        modelRewardTensor = torch.cat(modelRewardList)
        return modelRewardTensor

    def resetFixedEnvironments(self):
        for simEnvironment in self.simEnvironments:
            simEnvironment.resetFixed()

class SimEnvironment():
    def __init__(self, stage, device, envNumber, dbInputs=None,
                 globalLaparoscopeDragLookupRadius=0.15):
        
        self.simInteractions = None
        self.simMeshes = None
        self.simLockBoxes = None
        self.simConnectors = None
        self.simElasticRods = None
        self.simContext = None
        self.sqrtValues = None
        self.envNumber = envNumber
        self.stage = stage

        self.globalLaparoscopeDragLookupRadius = globalLaparoscopeDragLookupRadius

        if not dbInputs is None:
            self.simContext = SimContext()
            self.simContext.readFromDBInputs(dbInputs)
            self.simContext.setSimDevice(device)

        meshes = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Mesh) and x.GetAttribute("simMesh").Get() == True]
        lockBoxes = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Cube) and x.GetAttribute("simLockBox").Get() == True]
        connectors = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Xform) and x.GetAttribute("simConnector").Get() == True]
        elasticRods = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Xform) and x.GetAttribute("simElasticRod").Get() == True]
        laparoscopes = [x for x in self.stage.Traverse() if x.IsA(UsdGeom.Xform) and x.GetAttribute("simLaparoscope").Get() == True]
    
        self.simInteractions = SimInteractions()

        self.simMeshes = []
        for mesh in meshes:
            simMesh = dk.SimMesh(mesh, device)
            self.simMeshes.append(simMesh)
        
        self.simLockBoxes = []
        for lockBox in lockBoxes:
            simLockBox = dk.SimLockBox(lockBox, device)
            self.simLockBoxes.append(simLockBox)
    
        self.simConnectors = []
        for connector in connectors:
            simConnector = dk.SimConnector(connector, device)
            self.simConnectors.append(simConnector)
    
        self.simElasticRods = []
        for elasticRod in elasticRods:
            simElasticRod = dk.SimElasticRod(elasticRod, device)
            self.simElasticRods.append(simElasticRod)

        self.simLaparoscopes = []
        for laparoscope in laparoscopes:
            simLaparoscope = dk.SimLaparoscope(laparoscope, device, self.stage, laparoscopeDragLookupRadius = self.globalLaparoscopeDragLookupRadius)
            self.simLaparoscopes.append(simLaparoscope)
    
        for simMesh in self.simMeshes:
            for simLockBox in self.simLockBoxes:
                
                simLockBox.lockArray(simMesh.initialVertexHost.numpy(), simMesh.inverseMassHost.numpy())
                # newInverseMasses = simLockBox.lockArray(simMesh.initialVertexHost.numpy(), simMesh.inverseMassHost.numpy())
                simLockBox.lockArray(simMesh.initialVertexHost.numpy(), simMesh.inverseMassHost.numpy())
                # newInverseMassesNeoHookean = simLockBox.lockArray(simMesh.initialVertexHost.numpy(), simMesh.inverseMassHost.numpy())
                # simMesh.updateInverseMass(newInverseMasses)
                # simMesh.updateInverseMassNeoHookean(newInverseMassesNeoHookean)     
    
            for simConnector in self.simConnectors:
                if(simMesh.path == simConnector.pointBodyPath):
                    simConnector.updatePointInfo(simMesh)
                if(simMesh.path == simConnector.triBodyPath):
                    simConnector.updateTriInfo(simMesh)
    
        for simElasticRod in self.simElasticRods:
        
            for simLockBox in self.simLockBoxes:
                
                newInverseMasses = simLockBox.lockArray(simElasticRod.elementInitialVertex.numpy(), simElasticRod.elementInverseMassHost.numpy())
                simElasticRod.updateInverseMass(newInverseMasses)

        sqrtValues = np.array(np.sqrt([x*(1.0/in_de_crease_steps) for x in range(in_de_crease_steps + 1)]))
        self.sqrtValues = wp.array(sqrtValues, dtype=float, device=device)
    
    def copyInitialDataToPrims(self) -> None:
    
        for simMesh in self.simMeshes:
            holes = []
            holesNP = np.array(holes)

            #copy holes
            holesAtt = simMesh.prim.GetAttribute("holeIndices")
            holesAtt.Set(Vt.IntArray().FromNumpy(holesNP))

            # todo: Move to a function
            #copy points
            pointsAtt = simMesh.prim.GetAttribute("points")
            pointsHost = simMesh.initialVisPoint.numpy()
            pointsAtt.Set(Vt.Vec3fArray().FromNumpy(pointsHost))
    
        for simElasticRod in self.simElasticRods:
            # todo: Move to a function
            elementInitialVertexHost = simElasticRod.elementInitialVertex.numpy()
            elementInitialVertex = Vt.Vec3dArray().FromNumpy(elementInitialVertexHost)
            for i in range(len(simElasticRod.element)):
                element = simElasticRod.element[i]
                UsdGeom.XformCommonAPI(element).SetTranslate(elementInitialVertex[i])

    def removeOriginalMeshes(self):
        for simMesh in self.simMeshes:
            self.stage.RemovePrim(simMesh.path)

    def removeOriginalLaparoscopes(self):
        for simLaparoscope in self.simLaparoscopes:
            self.stage.RemovePrim(simLaparoscope.path)

    def duplicateMeshes(self, spacingX:float, spacingZ:float):

        xFormShift = ((self.envNumber//10)*spacingX, 0.00, (self.envNumber%10)*spacingZ)

        for simMesh in self.simMeshes:
            simMesh.name = simMesh.name + str(self.envNumber)
            simMesh.path = Sdf.Path("/World/" + simMesh.name)

            meshGeom = UsdGeom.Mesh.Define(self.stage, simMesh.path)
            meshGeom.CreatePointsAttr(simMesh.meshVisPoints)
            meshGeom.CreateFaceVertexCountsAttr([3]*simMesh.numTris)
            meshGeom.CreateFaceVertexIndicesAttr(simMesh.meshVisFaces)
            meshGeom.CreateNormalsAttr(simMesh.meshNormals)
            meshGeom.SetNormalsInterpolation("vertex")
            meshGeom.CreateSubdivisionSchemeAttr().Set("none")

            # Set position.
            UsdGeom.XformCommonAPI(meshGeom).SetTranslate(simMesh.translation + xFormShift)

            # Set rotation.
            UsdGeom.XformCommonAPI(meshGeom).SetRotate(simMesh.rotation, UsdGeom.XformCommonAPI.RotationOrderXYZ)

            # Set scale.
            UsdGeom.XformCommonAPI(meshGeom).SetScale((1.0, 1.0, 1.0))

            simMesh.prim = self.stage.GetPrimAtPath(simMesh.path)

    def duplicateLaparoscopes(self, spacingX:float, spacingZ:float):

        xFormShift = ((self.envNumber//10)*spacingX, 0.00, (self.envNumber%10)*spacingZ)

        # for simMesh in self.simMeshes:
        #     simMesh.name = simMesh.name + str(self.envNumber)
        #     simMesh.path = Sdf.Path("/World/" + simMesh.name)

        #     meshGeom = UsdGeom.Mesh.Define(self.stage, simMesh.path)
        #     meshGeom.CreatePointsAttr(simMesh.meshVisPoints)
        #     meshGeom.CreateFaceVertexCountsAttr([3]*simMesh.numTris)
        #     meshGeom.CreateFaceVertexIndicesAttr(simMesh.meshVisFaces)
        #     meshGeom.CreateNormalsAttr(simMesh.meshNormals)
        #     meshGeom.SetNormalsInterpolation("vertex")
        #     meshGeom.CreateSubdivisionSchemeAttr().Set("none")

        #     # Set position.
        #     UsdGeom.XformCommonAPI(meshGeom).SetTranslate(simMesh.translation + xFormShift)

        #     # Set rotation.
        #     UsdGeom.XformCommonAPI(meshGeom).SetRotate(simMesh.rotation, UsdGeom.XformCommonAPI.RotationOrderXYZ)

        #     # Set scale.
        #     UsdGeom.XformCommonAPI(meshGeom).SetScale((1.0, 1.0, 1.0))

        #     simMesh.prim = self.stage.GetPrimAtPath(simMesh.path)

    def applyActions(self, actions):
        for simLaparoscope in self.simLaparoscopes:
            simLaparoscope.applyActions(actions, self.simMeshes)

    def getEnvironmentObservationsTensor(self):
        environmentObservationsList = []
        for simMesh in self.simMeshes:
            environmentObservationsList.append(wp.to_torch(simMesh.vertex))
            environmentObservationsList.append(wp.to_torch(simMesh.velocity))
        
        for simLaparoscope in self.simLaparoscopes:
            environmentObservationsList.append(wp.to_torch(simLaparoscope.laparoscopeBase))
            environmentObservationsList.append(wp.to_torch(simLaparoscope.laparoscopeTip))

        environmentObservationsTensor = torch.cat(environmentObservationsList).flatten()
        return environmentObservationsTensor

    def getLaparoscopeObservationsTensor(self):
        laparoscopeObservationsList = []
        for simLaparoscope in self.simLaparoscopes:
            laparoscopeObservationsList.append(wp.to_torch(simLaparoscope.laparoscopeBase))
            laparoscopeObservationsList.append(wp.to_torch(simLaparoscope.laparoscopeTip))

        laparoscopeObservationsTensor = torch.cat(laparoscopeObservationsList).flatten()
        return laparoscopeObservationsTensor

    def getEnvironmentRewardTensor(self):
        environmentRewardList = []
        for simMesh in self.simMeshes:
            environmentRewardList.append(wp.to_torch(simMesh.vertex))

        environmentRewardTensor = torch.cat(environmentRewardList)
        return environmentRewardTensor

    def transformSimMeshData(self) -> None:
        for simMesh in self.simMeshes:
            simMesh.transformMeshData()

    def resetFixed(self):
        for simMesh in self.simMeshes:
            simMesh.resetFixed()

        for simLaparoscope in self.simLaparoscopes:
            simLaparoscope.resetFixed()