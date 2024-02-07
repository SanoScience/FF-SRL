import warp as wp
import FF_SRL as dk
import numpy as np


@wp.kernel
def calculateCentroids(vertex: wp.array(dtype=wp.vec3),
                       triangle: wp.array(dtype=wp.int32),
                       centroids: wp.array(dtype=wp.vec3)):
    
    tid = wp.tid()

    triA = triangle[tid * 3 + 0]
    triB = triangle[tid * 3 + 1]
    triC = triangle[tid * 3 + 2]

    pA = vertex[triA]
    pB = vertex[triB]
    pC = vertex[triC]

    centroids[tid] = (pA + pB + pC) / 3.0

@wp.kernel
def calculateTriIDs(triIDs: wp.array(dtype=wp.int32)):
    
    tid = wp.tid()

    triIDs[tid] = tid

@wp.kernel
def refitBVHLevel(vertex: wp.array(dtype=wp.vec3),
                  triangle: wp.array(dtype=wp.int32),
                  leftNodeId: wp.array(dtype=wp.int32),
                  triMap: wp.array(dtype=wp.int32),
                  firstTriId: wp.array(dtype=wp.int32),
                  triCount: wp.array(dtype=wp.int32),
                  level: wp.array(dtype=wp.int32),
                  aabbMin: wp.array(dtype=wp.vec3),
                  aabbMax: wp.array(dtype=wp.vec3),
                  currentLevel: wp.int32,
                  numNodes: wp.int32,
                  numEnv: wp.int32,
                  numEnvMeshVisFaces: wp.int32,
                  numEnvRigidVisFaces: wp.int32,
                  numEnvLaparoscopeVisFaces: wp.int32):
    
    tid = wp.tid()
    nodeId = tid % numNodes
    currentEnv = int(tid / numNodes)

    if not level[nodeId] == currentLevel:
        return
    
    leftNode = leftNodeId[nodeId]
    
    # If node has only triangles
    if leftNode < 0:

        firstTri = firstTriId[nodeId]

        nodeAabbMin = wp.vec3(1e6, 1e6, 1e6)
        nodeAabbMax = wp.vec3(-1e6, -1e6, -1e6)
        
        for nextTri in range(triCount[nodeId]):
        
            triId = triMap[firstTri + nextTri]
               
            # ToDo: This should be moved to a mapping array     
            visFaceShift = 0
            if triId > numEnvMeshVisFaces + numEnvRigidVisFaces:
                visFaceShift = (numEnv-1) * (numEnvMeshVisFaces + numEnvRigidVisFaces) + currentEnv * numEnvLaparoscopeVisFaces
            elif triId > numEnvMeshVisFaces:
                visFaceShift = (numEnv-1) * numEnvMeshVisFaces + currentEnv * numEnvRigidVisFaces
            else:
                visFaceShift = currentEnv * numEnvMeshVisFaces

            triId = triId + visFaceShift

            i = triangle[triId * 3]
            j = triangle[triId * 3 + 1]
            k = triangle[triId * 3 + 2]

            pA = vertex[i]
            pB = vertex[j]
            pC = vertex[k]

            nodeAabbMin = wp.min(pA, wp.min(pB, wp.min(pC, nodeAabbMin)))
            nodeAabbMax = wp.max(pA, wp.max(pB, wp.max(pC, nodeAabbMax)))

        aabbMin[tid] = nodeAabbMin
        aabbMax[tid] = nodeAabbMax

    # If node has children
    else:
        envShift = currentEnv * numNodes
        rightNode = leftNode + 1

        aabbMinLeft = aabbMin[leftNode + envShift]
        aabbMaxLeft = aabbMax[leftNode + envShift]
        aabbMinRight = aabbMin[rightNode + envShift]
        aabbMaxRight = aabbMax[rightNode + envShift]

        aabbMin[tid] = wp.min(aabbMinLeft, aabbMinRight)
        aabbMax[tid] = wp.max(aabbMaxLeft, aabbMaxRight)

class SimBVHNode():

    def __init__(self, id, leftNode=-1, firstTriId=None, triCount=None, aabbMin=None, aabbMax=None, level=-1):
        self.id = id
        self.aabbMin = aabbMin
        self.aabbMax = aabbMax
        self.leftNode = leftNode
        self.firstTriId = firstTriId
        self.triCount = triCount
        self.level = level

    def isLeaf(self) -> bool:
        return self.triCount > 0
    
class SimAABB():

    def __init__(self) -> None:
        self.bMin = np.array([1e30, 1e30, 1e30])
        self.bMax = np.array([-1e30, -1e30, -1e30])

    def growPoint(self, p):
        self.bMin = np.stack([self.bMin, p]).min(axis=0)
        self.bMax = np.stack([self.bMax, p]).max(axis=0)

    def growBounds(self, bound):
        self.bMin = np.stack([self.bMin, bound.bMin]).min(axis=0)
        self.bMax = np.stack([self.bMax, bound.bMax]).max(axis=0)

    def area(self) -> float:
        extent = self.bMax - self.bMin
        return extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0]

class SimBin():
    
    def __init__(self):
        self.bounds = SimAABB()
        self.triCount = 0

class SimBVH():
    # Implementation of Bounding Volume Hierarchy based on
    # https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

    def __init__(self, device, simModel:dk.SimModelDO, binsNumber:int=8, save:bool=False, load:bool=False, path="bvh.npz") -> None:
        
        self.device = device
        self.simModel = simModel
        self.binsNumber = binsNumber
        self.refitGraph = None
        self.nodes = []
        self.numEnvs = self.simModel.numEnvs

        if load:
            self.loadBVH(path)
        else:
            self.buildBVH()

        if save:
            self.saveBVH(path)

    def saveBVH(self, path):

        np.savez(path,
                 self.triId.numpy(),
                 self.aabbMin.numpy()[:self.numNodes],
                 self.aabbMax.numpy()[:self.numNodes],
                 self.leftNode.numpy(),
                 self.firstTriId.numpy(),
                 self.triCount.numpy(),
                 self.level.numpy())

    def loadBVH(self, path):
        
        loadedArrays = np.load(path)

        self.triId = wp.array(loadedArrays['arr_0'], dtype=wp.int32, device=self.device)
        aabbMin = loadedArrays['arr_1']
        aabbMax = loadedArrays['arr_2']
        self.leftNode = wp.array(loadedArrays['arr_3'], dtype=wp.int32, device=self.device)
        self.firstTriId = wp.array(loadedArrays['arr_4'], dtype=wp.int32, device=self.device)
        self.triCount = wp.array(loadedArrays['arr_5'], dtype=wp.int32, device=self.device)
        self.level = wp.array(loadedArrays['arr_6'], dtype=wp.int32, device=self.device)

        self.numNodes = len(aabbMin)

        # Environments will only differ with the values of aabb boxes
        aabbMin = np.repeat(aabbMin, self.numEnvs)
        aabbMax = np.repeat(aabbMax, self.numEnvs)
        self.aabbMin = wp.array(aabbMin, dtype=wp.vec3, device=self.device)
        self.aabbMax = wp.array(aabbMax, dtype=wp.vec3, device=self.device)

        self.numAllNodes = self.numNodes * self.numEnvs
        self.maxLevel = self.level.numpy().max()

    def getTreeDataRecur(self, output, nodeId, col, row, height):

        if height < 1:
            return

        node = self.nodes[nodeId]
        output[row][col] = node.firstTriId
        if node.leftNode > -1:
            self.getTreeDataRecur(output, node.leftNode, col - pow(2, height - 2), row + 1, height - 1)
            self.getTreeDataRecur(output, node.leftNode + 1, col + pow(2, height - 2), row + 1, height - 1)

    def printBVH(self):

        maxLevelHeight = self.maxLevel + 1
        maxLevelWidth = pow(2, maxLevelHeight)

        outputBuffer = [[-1]*maxLevelWidth for i in range(maxLevelHeight)]

        self.getTreeDataRecur(outputBuffer, 0, int(maxLevelWidth/2), 0, maxLevelHeight)

        # for i in range(self.maxLevel + 1):
        for i in range(maxLevelHeight):
            for j in range(maxLevelWidth):
                if outputBuffer[i][j] == -1:
                    print(' ', end = ' ')
                else:
                    print(outputBuffer[i][j], end = ' ')
            print("\n")

    def evaluateSAH(self, node, axis:int, pos):
        leftBox = SimAABB()
        rightBox = SimAABB()
        leftCount = 0
        rightCount = 0

        for i in range(node.triCount):

            triId = self.triIdHost[node.firstTriId + i]

            centroid = self.centroidsHost[triId][axis]

            triA = self.triangleHost[triId * 3 + 0]
            triB = self.triangleHost[triId * 3 + 1]
            triC = self.triangleHost[triId * 3 + 2]

            pA = self.vertexHost[triA]
            pB = self.vertexHost[triB]
            pC = self.vertexHost[triC]

            if centroid < pos:
                leftCount += 1
                leftBox.growPoint(pA)
                leftBox.growPoint(pB)
                leftBox.growPoint(pC)
            else:
                rightCount += 1
                rightBox.growPoint(pA)
                rightBox.growPoint(pB)
                rightBox.growPoint(pC)

        cost = leftCount * leftBox.area() + rightCount * rightBox.area()
        return cost if cost > 0 else 1e30
    
    def findBestSplitPlane(self, nodeId):

        node = self.nodes[nodeId]

        bestCost= 1e30
        bestAxis = -1
        bestPos = 0.0

        for axis in range(3):
            for j in range(node.triCount):
                candidatePos = self.centroidsHost[self.triIdHost[node.firstTriId + j]][axis]
                cost = self.evaluateSAH(node, axis, candidatePos)
                if cost < bestCost:
                    bestPos = candidatePos
                    bestAxis = axis
                    bestCost = cost

        return bestCost, bestAxis, bestPos
    
    def findBestSplitPlaneBinned(self, nodeId):

        node = self.nodes[nodeId]

        bestCost= 1e30
        bestAxis = -1
        bestPos = 0.0

        for axis in range(3):

            boundsMin = 1e30
            boundsMax = -1e30

            for j in range(node.triCount):
                candidatePos = self.centroidsHost[self.triIdHost[node.firstTriId + j]][axis]
                boundsMin = wp.min(boundsMin, candidatePos)
                boundsMax = wp.max(boundsMax, candidatePos)

            if boundsMax == boundsMin:
                continue

            bins = [SimBin() for i in range(self.binsNumber)]
            scale = self.binsNumber / (boundsMax - boundsMin)

            for j in range(node.triCount):
                triId = self.triIdHost[node.firstTriId + j]
                candidatePos = self.centroidsHost[triId][axis]
                binId = int(wp.min(self.binsNumber - 1, int((candidatePos - boundsMin) * scale)))
                bins[binId].triCount += 1
                bins[binId].bounds.growPoint(self.vertexHost[self.triangleHost[triId * 3 + 0]])
                bins[binId].bounds.growPoint(self.vertexHost[self.triangleHost[triId * 3 + 1]])
                bins[binId].bounds.growPoint(self.vertexHost[self.triangleHost[triId * 3 + 2]])

            leftArea = [0.0] * (self.binsNumber - 1)
            rightArea = [0.0] * (self.binsNumber - 1)
            leftCount = [0] * (self.binsNumber - 1)
            rightCount = [0] * (self.binsNumber - 1)
            leftBox = SimAABB()
            rightBox = SimAABB()
            leftSum = 0
            rightSum = 0

            for j in range(self.binsNumber - 1):
                leftSum += bins[j].triCount
                leftCount[j] = leftSum
                leftBox.growBounds(bins[j].bounds)
                leftArea[j] = leftBox.area()

                rightSum += bins[self.binsNumber - 1 - j].triCount
                rightCount[self.binsNumber - 2 - j] = rightSum
                rightBox.growBounds(bins[self.binsNumber - 1 - j].bounds)
                rightArea[self.binsNumber - 2 - j] = rightBox.area()

            scale = (boundsMax - boundsMin) / self.binsNumber
            for j in range(self.binsNumber - 1):
                planeCost = leftCount[j] * leftArea[j] + rightCount[j] * rightArea[j]
                if planeCost < bestCost:
                    bestAxis = axis
                    bestPos = boundsMin + scale * (j + 1)
                    bestCost = planeCost

        return bestCost, bestAxis, bestPos

    def calculateNodeCost(self, nodeId):

        node = self.nodes[nodeId]

        extent = node.aabbMax - node.aabbMin
        parentArea = extent[0] * extent[1] + extent[1] * extent[2] + extent[2] * extent[0]
        parentCost = node.triCount * parentArea

        return parentCost

    def subdivide(self, nodeId, level):

        node = self.nodes[nodeId]
        
        # determine split axis and position
        # bestCost, bestAxis, bestPos = self.findBestSplitPlane(nodeId)
        bestCost, bestAxis, bestPos = self.findBestSplitPlaneBinned(nodeId)
        
        parentCost = self.calculateNodeCost(nodeId)

        if bestCost >= parentCost:
            return

        # in-place partition
        i = node.firstTriId
        j = i + node.triCount - 1
        
        while i <= j:
            if (self.centroidsHost[self.triIdHost[i]][bestAxis] < bestPos):
                i += 1
            else:
                self.triIdHost[i], self.triIdHost[j] = self.triIdHost[j], self.triIdHost[i]
                j -= 1

        leftCount = i - node.firstTriId
        if (leftCount == 0 or leftCount == node.triCount):
            return
        
        leftChildIdx = self.nodesUsed
        rightChildIdx = self.nodesUsed + 1
        self.nodesUsed += 2

        self.nodes[leftChildIdx].firstTriId = node.firstTriId
        self.nodes[leftChildIdx].triCount = leftCount
        self.nodes[leftChildIdx].level = level + 1
        self.nodes[rightChildIdx].firstTriId = i
        self.nodes[rightChildIdx].triCount = node.triCount - leftCount
        self.nodes[rightChildIdx].level = level + 1

        node.leftNode = leftChildIdx
        node.triCount = 0

        self.updateNodeBounds(leftChildIdx)
        self.updateNodeBounds(rightChildIdx)

        # recurse
        self.subdivide(leftChildIdx, level + 1)
        self.subdivide(rightChildIdx, level + 1)

    def updateNodeBounds(self, nodeId):
        
        node = self.nodes[nodeId]
        node.aabbMin = np.array([1e30, 1e30, 1e30])
        node.aabbMax = np.array([-1e30, -1e30, -1e30])

        for i in range(node.firstTriId, node.firstTriId + node.triCount):
            triId = self.triIdHost[i]

            triA = self.triangleHost[triId * 3 + 0]
            triB = self.triangleHost[triId * 3 + 1]
            triC = self.triangleHost[triId * 3 + 2]
            
            node.aabbMin = np.stack([node.aabbMin, self.vertexHost[triA], self.vertexHost[triB], self.vertexHost[triC]]).min(axis=0)
            node.aabbMax = np.stack([node.aabbMax, self.vertexHost[triA], self.vertexHost[triB], self.vertexHost[triC]]).max(axis=0)

    def refitBVHLoop(self):

        for i in range(self.maxLevel + 1):
            wp.launch(kernel=refitBVHLevel,
                      dim=self.numAllNodes,
                      inputs=[
                          self.simModel.allVisPoint,
                          self.simModel.allVisFace,
                          self.leftNode,
                          self.triId,
                          self.firstTriId,
                          self.triCount,
                          self.level,
                          self.aabbMin,
                          self.aabbMax,
                          self.maxLevel - i,
                          self.numNodes,
                          self.simModel.numEnvs,
                          self.simModel.numEnvMeshesVisFaces,
                          self.simModel.numEnvRigidsVisFaces,
                          self.simModel.numEnvLaparoscopeVisFaces
                      ])

    def refitBVH(self, useGraph=False):

        if useGraph:

            if self.refitGraph == None:
                wp.capture_begin()
                self.refitBVHLoop()
                self.refitGraph = wp.capture_end()

            wp.capture_launch(self.refitGraph)
        
        else:
            self.refitBVHLoop()

    def buildBVH(self):

        self.nodes = [SimBVHNode(i) for i in range(self.simModel.numEnvAllVisFaces * 2)]
        self.centroids = wp.zeros(self.simModel.numEnvAllVisFaces, dtype=wp.vec3, device=self.device)
        self.triId = wp.zeros(self.simModel.numEnvAllVisFaces, dtype=wp.int32, device=self.device)

        wp.launch(kernel=calculateCentroids, 
          dim=self.simModel.numEnvAllVisFaces, 
          inputs=[self.simModel.envVisPoint,
                  self.simModel.envVisFace,
                  self.centroids],
          device=self.device)

        wp.launch(kernel=calculateTriIDs, 
          dim=self.simModel.numEnvAllVisFaces, 
          inputs=[self.triId],
          device=self.device)
        
        self.vertexHost = self.simModel.envVisPoint.numpy()
        self.triangleHost = self.simModel.envVisFace.numpy()
        self.centroidsHost = self.centroids.numpy()
        self.triIdHost = self.triId.numpy()

        level = 0
        self.nodes[0] = SimBVHNode(0, 0, 0, self.simModel.numEnvAllVisFaces, self.vertexHost.min(axis=0), self.vertexHost.max(axis=0), 0)
        self.nodesUsed = 1
        self.subdivide(0, level)

        aabbMin = []
        aabbMax = []
        leftNode = []
        firstTri = []
        triCount = []
        level = []

        maxLevel = 0
        for node in self.nodes:
            if not node.aabbMin is None:
                aabbMin.append(node.aabbMin)
                aabbMax.append(node.aabbMax)
                leftNode.append(node.leftNode)
                firstTri.append(node.firstTriId)
                triCount.append(node.triCount)
                level.append(node.level)

                if node.level > maxLevel:
                    maxLevel = node.level

        self.numNodes = len(aabbMin)

        aabbMin = aabbMin * self.numEnvs
        aabbMax = aabbMax * self.numEnvs

        self.numAllNodes = self.numNodes * self.numEnvs
        self.maxLevel = maxLevel

        self.triId = wp.array(self.triIdHost, dtype=wp.int32, device=self.device)
        self.aabbMin = wp.array(aabbMin, dtype=wp.vec3, device=self.device)
        self.aabbMax = wp.array(aabbMax, dtype=wp.vec3, device=self.device)
        self.leftNode = wp.array(leftNode, dtype=wp.int32, device=self.device)
        self.firstTriId = wp.array(firstTri, dtype=wp.int32, device=self.device)
        self.triCount = wp.array(triCount, dtype=wp.int32, device=self.device)
        self.level = wp.array(level, dtype=wp.int32, device=self.device)

