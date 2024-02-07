import warp as wp
import math
import sys
import numpy as np
import FF_SRL as dk
from pxr import Usd, UsdGeom, Gf

FLOAT_EPSILON = wp.constant(sys.float_info.epsilon)
in_de_crease_steps = 5
IN_DE_CREASE_STEPS = wp.constant(in_de_crease_steps)

@wp.kernel
def bounds(predictedVertex: wp.array(dtype=wp.vec3),
           groundLevel: float):
    
    tid = wp.tid()

    x = predictedVertex[tid]

    if(x[1] < groundLevel):
        predictedVertex[tid] = wp.vec3(x[0], groundLevel, x[2])

@wp.kernel
def PBDStep(vertex: wp.array(dtype=wp.vec3),
            predictedVertex: wp.array(dtype=wp.vec3),
            velocity: wp.array(dtype=wp.vec3),
            dT: float):
    
    tid = wp.tid()

    x = vertex[tid]
    xPred = predictedVertex[tid]

    v = (xPred - x)*(1.0/dT)
    x = xPred

    vertex[tid] = x
    velocity[tid] = v

@wp.kernel
def gravity(vertex: wp.array(dtype=wp.vec3),
            predictedVertex: wp.array(dtype=wp.vec3),
            velocity: wp.array(dtype=wp.vec3),
            inverseMass: wp.array(dtype=float),
            gravityConstant: wp.vec3,
            velocityDampening: float,
            dt: float):
    
    tid = wp.tid()

    x = vertex[tid]
    v = velocity[tid]
    invMass = inverseMass[tid]

    velocityDampening = 1.0 - velocityDampening

    v = v + gravityConstant*invMass*dt*velocityDampening

    if (invMass < FLOAT_EPSILON):
        v = wp.vec3(0.0, 0.0, 0.0)

    xPred = x + v*dt
    predictedVertex[tid] = xPred

@wp.kernel
def distanceConstraints(predictedVertex: wp.array(dtype=wp.vec3),
                        dP: wp.array(dtype=wp.vec3),
                        constraintsNumber: wp.array(dtype=int),
                        edge: wp.array(dtype=int),
                        edgeRestLength: wp.array(dtype=float),
                        inverseMass: wp.array(dtype=float),
                        activeEdge: wp.array(dtype=float),
                        kS: float):
    
    tid = wp.tid()
    active = activeEdge[tid]

    if(active == 0.0):
        return

    edgeIndexA = edge[tid * 2]
    edgeIndexB = edge[tid * 2 + 1]

    edgePositionA = predictedVertex[edgeIndexA]
    edgePositionB = predictedVertex[edgeIndexB]
    
    edgeRestLen = edgeRestLength[tid]

    if (edgeRestLen < FLOAT_EPSILON):
        return

    dir = edgePositionB - edgePositionA
    len = wp.length(dir)

    if (len < FLOAT_EPSILON):
        return

    invMass = inverseMass[edgeIndexA] + inverseMass[edgeIndexB]
    if (invMass < FLOAT_EPSILON):
        return

    C = len-edgeRestLen
    edgeDP = -C * (dir / len) * kS * active / invMass

    wp.atomic_sub(dP, edgeIndexA, edgeDP * inverseMass[edgeIndexA])
    wp.atomic_add(dP, edgeIndexB, edgeDP * inverseMass[edgeIndexB])

    wp.atomic_add(constraintsNumber, edgeIndexA, 1)
    wp.atomic_add(constraintsNumber, edgeIndexB, 1)

@wp.kernel
def distanceConstraintsIterOdd(predictedVertex: wp.array(dtype=wp.vec3),
                               edge: wp.array(dtype=int),
                               edgeRestLength: wp.array(dtype=float),
                               inverseMass: wp.array(dtype=float),
                               activeEdge: wp.array(dtype=float),
                               kS: float):
    
    tid = wp.tid()
    active = activeEdge[tid]

    if(active == 0.0):
        return

    edgeIndexA = edge[tid * 4]
    edgeIndexB = edge[tid * 4 + 1]

    edgePositionA = predictedVertex[edgeIndexA]
    edgePositionB = predictedVertex[edgeIndexB]
    
    edgeRestLen = edgeRestLength[tid]
    if (edgeRestLen < FLOAT_EPSILON):
        return

    dir = edgePositionB - edgePositionA
    len = wp.length(dir)

    if (len < FLOAT_EPSILON):
        return

    invMass = inverseMass[edgeIndexA] + inverseMass[edgeIndexB]
    if (invMass < FLOAT_EPSILON):
        return

    C = len-edgeRestLen
    edgeDP = -C * (dir / len) * kS * active / invMass
    
    predictedVertex[edgeIndexA] = edgePositionA - edgeDP * inverseMass[edgeIndexA]
    predictedVertex[edgeIndexB] = edgePositionB + edgeDP * inverseMass[edgeIndexB]

@wp.kernel
def distanceConstraintsIterEven(predictedVertex: wp.array(dtype=wp.vec3),
                                edge: wp.array(dtype=int),
                                edgeRestLength: wp.array(dtype=float),
                                inverseMass: wp.array(dtype=float),
                                activeEdge: wp.array(dtype=float),
                                kS: float):
    
    tid = wp.tid()
    active = activeEdge[tid]

    if(active == 0.0):
        return

    edgeIndexA = edge[tid * 4 + 2]
    edgeIndexB = edge[tid * 4 + 3]

    edgePositionA = predictedVertex[edgeIndexA]
    edgePositionB = predictedVertex[edgeIndexB]
    
    edgeRestLen = edgeRestLength[tid]
    if (edgeRestLen < FLOAT_EPSILON):
        return

    dir = edgePositionB - edgePositionA
    len = wp.length(dir)

    if (len < FLOAT_EPSILON):
        return

    invMass = inverseMass[edgeIndexA] + inverseMass[edgeIndexB]
    if (invMass < FLOAT_EPSILON):
        return

    C = len-edgeRestLen
    edgeDP = -C * (dir / len) * kS * active / invMass
    
    predictedVertex[edgeIndexA] = edgePositionA - edgeDP * inverseMass[edgeIndexA]
    predictedVertex[edgeIndexB] = edgePositionB + edgeDP * inverseMass[edgeIndexB]

@wp.kernel
def distanceConstraintsXPBD(predictedVertex: wp.array(dtype=wp.vec3),
                        dP: wp.array(dtype=wp.vec3),
                        lambdas: wp.array(dtype=float),
                        constraintsNumber: wp.array(dtype=int),
                        edgeA: wp.array(dtype=int),
                        edgeB: wp.array(dtype=int),
                        edgeRestLength: wp.array(dtype=float),
                        inverseMass: wp.array(dtype=float),
                        activeEdge: wp.array(dtype=float),
                        compliance: float):
    
    tid = wp.tid()

    if(activeEdge[tid] == 0.0):
        return

    edgeIndexA = edgeA[tid]
    edgeIndexB = edgeB[tid]

    edgePositionA = predictedVertex[edgeIndexA]
    edgePositionB = predictedVertex[edgeIndexB]
    
    edgeRestLen = edgeRestLength[tid]
    if (edgeRestLen < FLOAT_EPSILON):
        return

    edgeLambda = lambdas[tid]

    dir = edgePositionB - edgePositionA
    len = wp.length(dir)

    if (len < FLOAT_EPSILON):
        return

    invMass = inverseMass[edgeIndexA] + inverseMass[edgeIndexB]
    if (invMass < FLOAT_EPSILON):
        return

    C = len-edgeRestLen

    dLambda = -(C + compliance * edgeLambda) / (invMass + compliance)
    lambdas[tid] = edgeLambda + dLambda
    edgeDP = dLambda * wp.normalize(dir)

    wp.atomic_sub(dP, edgeIndexA, edgeDP * inverseMass[edgeIndexA])
    wp.atomic_add(dP, edgeIndexB, edgeDP * inverseMass[edgeIndexB])

    wp.atomic_add(constraintsNumber, edgeIndexA, 1)
    wp.atomic_add(constraintsNumber, edgeIndexB, 1)

@wp.kernel
def triToPointDistanceConstraints(predictedVertexPoint: wp.array(dtype=wp.vec3),
                                  predictedVertexTri: wp.array(dtype=wp.vec3),
                                  dPPoint: wp.array(dtype=wp.vec3),
                                  dPTri: wp.array(dtype=wp.vec3),
                                  constraintsNumberPoint: wp.array(dtype=int),
                                  constraintsNumberTri: wp.array(dtype=int),
                                  triIds: wp.array(dtype=int),
                                  triBar: wp.array(dtype=wp.vec3),
                                  pointIds: wp.array(dtype=int),
                                  restDist: wp.array(dtype=float),
                                  massWeightRatio: float,
                                  kS: float,
                                  restLengthMul: float):
    
    tid = wp.tid()

    triIdA = triIds[tid * 3]
    triIdB = triIds[tid * 3 + 1]
    triIdC = triIds[tid * 3 + 2]

    triAPos = predictedVertexTri[triIdA]
    triBPos = predictedVertexTri[triIdB]
    triCPos = predictedVertexTri[triIdC]

    triBarPos = triBar[tid]

    triCenter = triAPos * triBarPos[0] + triBPos * triBarPos[1] + triCPos * triBarPos[2]

    pointId = pointIds[tid]
    pointPos = predictedVertexPoint[pointId]

    restDistPos = restDist[tid]

    dir = pointPos - triCenter
    len = wp.length(dir)
        
    if (len < FLOAT_EPSILON):
        return

    invMassPoint = massWeightRatio
    invMassTri = 1.0 - massWeightRatio

    C = len - restDistPos * restLengthMul
    s = invMassPoint + invMassTri * (triBarPos[0] * triBarPos[0] + triBarPos[1] * triBarPos[1] + triBarPos[2] * triBarPos[2])

    dP = (C / s) * (dir / len) * kS

    wp.atomic_sub(dPPoint, pointId, dP * invMassPoint)
    wp.atomic_add(dPTri, triIdA, dP * invMassTri * triBarPos[0])
    wp.atomic_add(dPTri, triIdB, dP * invMassTri * triBarPos[1])
    wp.atomic_add(dPTri, triIdC, dP * invMassTri * triBarPos[2])

    wp.atomic_add(constraintsNumberPoint, pointId, 1)
    wp.atomic_add(constraintsNumberTri, triIdA, 1)
    wp.atomic_add(constraintsNumberTri, triIdB, 1)
    wp.atomic_add(constraintsNumberTri, triIdC, 1)

@wp.kernel
def triToPointDistanceConstraintsDO(predictedVertex: wp.array(dtype=wp.vec3),
                                    dP: wp.array(dtype=wp.vec3),
                                    constraintsNumber: wp.array(dtype=int),
                                    triIds: wp.array(dtype=int),
                                    triBar: wp.array(dtype=wp.vec3),
                                    pointIds: wp.array(dtype=int),
                                    restDist: wp.array(dtype=float),
                                    massWeightRatio: float,
                                    kS: float,
                                    restLengthMul: float):
    
    tid = wp.tid()

    triIdA = triIds[tid * 3]
    triIdB = triIds[tid * 3 + 1]
    triIdC = triIds[tid * 3 + 2]

    triAPos = predictedVertex[triIdA]
    triBPos = predictedVertex[triIdB]
    triCPos = predictedVertex[triIdC]

    triBarPos = triBar[tid]

    triCenter = triAPos * triBarPos[0] + triBPos * triBarPos[1] + triCPos * triBarPos[2]

    pointId = pointIds[tid]
    pointPos = predictedVertex[pointId]

    restDistPos = restDist[tid]

    dir = pointPos - triCenter
    len = wp.length(dir)
        
    if (len < FLOAT_EPSILON):
        return

    invMassPoint = massWeightRatio
    invMassTri = 1.0 - massWeightRatio

    C = len - restDistPos * restLengthMul
    s = invMassPoint + invMassTri * (triBarPos[0] * triBarPos[0] + triBarPos[1] * triBarPos[1] + triBarPos[2] * triBarPos[2])

    delta = (C / s) * (dir / len) * kS

    wp.atomic_sub(dP, pointId, delta * invMassPoint)
    wp.atomic_add(dP, triIdA, delta * invMassTri * triBarPos[0])
    wp.atomic_add(dP, triIdB, delta * invMassTri * triBarPos[1])
    wp.atomic_add(dP, triIdC, delta * invMassTri * triBarPos[2])

    wp.atomic_add(constraintsNumber, pointId, 1)
    wp.atomic_add(constraintsNumber, triIdA, 1)
    wp.atomic_add(constraintsNumber, triIdB, 1)
    wp.atomic_add(constraintsNumber, triIdC, 1)

@wp.kernel
def dragConstraintsDP(predictedVertex: wp.array(dtype=wp.vec3),
                      dP: wp.array(dtype=wp.vec3),
                      lambdas: wp.array(dtype=float),
                      constraintsNumber: wp.array(dtype=int),
                      dragActive: wp.array(dtype=float),
                      laparoscopeInfo: wp.array(dtype = wp.vec3),
                      dragKs: float):
    
    tid = wp.tid()
    drag = dragActive[tid]

    if drag == 0.0:
        return

    predictedPosition = predictedVertex[tid]
    laparoscopeDragPoint = laparoscopeInfo[3]

    dir = predictedPosition - laparoscopeDragPoint
    len = wp.length(dir)

    if (len < FLOAT_EPSILON):
        return

    invMass = 2.0

    edgeDP = len * wp.normalize(dir) * drag * dragKs / invMass
    
    wp.atomic_sub(dP, tid, edgeDP)
    wp.atomic_add(constraintsNumber, tid, 1)

@wp.kernel
def dragConstraints(predictedVertex: wp.array(dtype=wp.vec3),
                    dragActive: wp.array(dtype=float),
                    laparoscopeInfo: wp.array(dtype = wp.vec3),
                    dragKs: float):
    tid = wp.tid()
    drag = dragActive[tid]

    if drag == 0.0:
        return

    predictedPosition = predictedVertex[tid]
    laparoscopeDragPoint = laparoscopeInfo[3]

    dir = predictedPosition - laparoscopeDragPoint
    len = wp.length(dir)

    if (len < FLOAT_EPSILON):
        return

    invMass = 2.0

    edgeDP = len * wp.normalize(dir) * drag * dragKs / invMass

    wp.atomic_sub(predictedVertex, tid, edgeDP)

@wp.kernel
def dragConstraintsDO(predictedVertex: wp.array(dtype=wp.vec3),
                      dragActive: wp.array(dtype=float),
                      vertexToEnv: wp.array(dtype=wp.int32),
                      laparoscopeInfo: wp.array(dtype = wp.vec3),
                      dragKs: float):
    
    tid = wp.tid()
    drag = dragActive[tid]
    numEnv = vertexToEnv[tid]

    if drag == 0.0:
        return

    predictedPosition = predictedVertex[tid]
    # laparoscopeDragPoint = laparoscopeInfo[numEnv * 4 + 3]
    # In DVRK we use right clamp end for dragging
    laparoscopeDragPoint = laparoscopeInfo[numEnv * 4 + 2]

    dir = predictedPosition - laparoscopeDragPoint
    len = wp.length(dir)

    if (len < FLOAT_EPSILON):
        return

    invMass = 2.0

    edgeDP = len * wp.normalize(dir) * drag * dragKs / invMass

    # wp.atomic_sub(predictedVertex, tid, edgeDP)
    predictedVertex[tid] = predictedPosition - edgeDP

@wp.kernel
def volumeConstraints(predictedVertex: wp.array(dtype=wp.vec3),
                      dP: wp.array(dtype=wp.vec3),
                      constraintsNumber: wp.array(dtype=wp.int32),
                      tetrahedron: wp.array(dtype=wp.int32),
                      tetrahedronRestVolume: wp.array(dtype=wp.float32),
                      inverseMass: wp.array(dtype=wp.float32),
                      activeTetrahedron: wp.array(dtype=wp.float32),
                      kS: float):
    
    tid = wp.tid()
    active = activeTetrahedron[tid]

    if not active == 1.0:
        return

    tetrahedronIndexA = tetrahedron[tid * 4]
    tetrahedronIndexB = tetrahedron[tid * 4 + 1]
    tetrahedronIndexC = tetrahedron[tid * 4 + 2]
    tetrahedronIndexD = tetrahedron[tid * 4 + 3]

    tetrahedronPositionA = predictedVertex[tetrahedronIndexA]
    tetrahedronPositionB = predictedVertex[tetrahedronIndexB]
    tetrahedronPositionC = predictedVertex[tetrahedronIndexC]
    tetrahedronPositionD = predictedVertex[tetrahedronIndexD]
    
    tetrahedronRestVol = tetrahedronRestVolume[tid]

    p1 = tetrahedronPositionB - tetrahedronPositionA
    p2 = tetrahedronPositionC - tetrahedronPositionA
    p3 = tetrahedronPositionD - tetrahedronPositionA

    q2 = wp.cross(p3, p1)
    q1 = wp.cross(p2, p3)
    q3 = wp.cross(p1, p2)
    q0 = - q1 - q2 - q3

    mA = inverseMass[tetrahedronIndexA]
    mB = inverseMass[tetrahedronIndexB]
    mC = inverseMass[tetrahedronIndexC]
    mD = inverseMass[tetrahedronIndexD]

    volume = wp.abs(wp.dot(wp.cross(p1, p2), p3)) / 6.0

    w = mA * wp.dot(q0, q0) + mB * wp.dot(q1, q1) + mC * wp.dot(q2, q2) + mD * wp.dot(q3, q3)

    if(wp.abs(w) < FLOAT_EPSILON):
        return

    C = tetrahedronRestVol - volume
    dLambda = - kS * active * C / w

    wp.atomic_add(dP, tetrahedronIndexA, q0 * dLambda * mA)
    wp.atomic_add(dP, tetrahedronIndexB, q1 * dLambda * mB)
    wp.atomic_add(dP, tetrahedronIndexC, q2 * dLambda * mC)
    wp.atomic_add(dP, tetrahedronIndexD, q3 * dLambda * mD)

    wp.atomic_add(constraintsNumber, tetrahedronIndexA, 1)
    wp.atomic_add(constraintsNumber, tetrahedronIndexB, 1)
    wp.atomic_add(constraintsNumber, tetrahedronIndexC, 1)
    wp.atomic_add(constraintsNumber, tetrahedronIndexD, 1)


@wp.kernel
def volumeConstraintsXPBD(predictedVertex: wp.array(dtype=wp.vec3),
                      dP: wp.array(dtype=wp.vec3),
                      lambdas: wp.array(dtype=float),
                      constraintsNumber: wp.array(dtype=int),
                      tetrahedronA: wp.array(dtype=int),
                      tetrahedronB: wp.array(dtype=int),
                      tetrahedronC: wp.array(dtype=int),
                      tetrahedronD: wp.array(dtype=int),
                      tetrahedronRestVolume: wp.array(dtype=float),
                      inverseMass: wp.array(dtype=float),
                      activeTetrahedron: wp.array(dtype=float),
                      alpha: float):
    
    tid = wp.tid()

    if not activeTetrahedron[tid] == 1.0:
        return

    tetrahedronIndexA = tetrahedronA[tid]
    tetrahedronIndexB = tetrahedronB[tid]
    tetrahedronIndexC = tetrahedronC[tid]
    tetrahedronIndexD = tetrahedronD[tid]

    tetrahedronPositionA = predictedVertex[tetrahedronIndexA]
    tetrahedronPositionB = predictedVertex[tetrahedronIndexB]
    tetrahedronPositionC = predictedVertex[tetrahedronIndexC]
    tetrahedronPositionD = predictedVertex[tetrahedronIndexD]
    
    tetrahedronRestVol = tetrahedronRestVolume[tid]

    tetrahedronLambda = lambdas[tid]

    p1 = tetrahedronPositionB - tetrahedronPositionA
    p2 = tetrahedronPositionC - tetrahedronPositionA
    p3 = tetrahedronPositionD - tetrahedronPositionA

    q2 = wp.cross(p3, p1)
    q1 = wp.cross(p2, p3)
    q3 = wp.cross(p1, p2)
    q0 = - q1 - q2 - q3

    mA = inverseMass[tetrahedronIndexA]
    mB = inverseMass[tetrahedronIndexB]
    mC = inverseMass[tetrahedronIndexC]
    mD = inverseMass[tetrahedronIndexD]

    volume = wp.dot(wp.cross(p1, p2), p3) / 6.0

    w = mA * wp.dot(q0, q0) + mB * wp.dot(q1, q1) + mC * wp.dot(q2, q2) + mD * wp.dot(q3, q3)

    if(wp.abs(w) < FLOAT_EPSILON):
        return

    C = volume - tetrahedronRestVol
    dLambda = -(C + alpha * tetrahedronLambda) / (w + alpha)
    # dLambda = - C / (w + alpha)
    # dLambda = - kS * C / w

    lambdas[tid] = tetrahedronLambda + dLambda

    wp.atomic_add(dP, tetrahedronIndexA, q0 * dLambda * mA)
    wp.atomic_add(dP, tetrahedronIndexB, q1 * dLambda * mB)
    wp.atomic_add(dP, tetrahedronIndexC, q2 * dLambda * mC)
    wp.atomic_add(dP, tetrahedronIndexD, q3 * dLambda * mD)

    wp.atomic_add(constraintsNumber, tetrahedronIndexA, 1)
    wp.atomic_add(constraintsNumber, tetrahedronIndexB, 1)
    wp.atomic_add(constraintsNumber, tetrahedronIndexC, 1)
    wp.atomic_add(constraintsNumber, tetrahedronIndexD, 1)

@wp.kernel
def neoHookeanConstraints(predictedVertex: wp.array(dtype=wp.vec3),
                      dP: wp.array(dtype=wp.vec3),
                      lambdas: wp.array(dtype=float),
                      constraintsNumber: wp.array(dtype=int),
                      tetrahedronA: wp.array(dtype=int),
                      tetrahedronB: wp.array(dtype=int),
                      tetrahedronC: wp.array(dtype=int),
                      tetrahedronD: wp.array(dtype=int),
                      tetrahedronRestVolume: wp.array(dtype=float),
                      inverseMass: wp.array(dtype=float),
                      inverseRestPositions: wp.array(dtype=wp.mat33),
                      volumeCompliance: float,
                      deviatoricCompliance: float,
                      dT: float):
    
    tid = wp.tid()

    restVolume = tetrahedronRestVolume[tid]
    lmbd = lambdas[tid]

    tetrahedronIndexA = tetrahedronA[tid]
    tetrahedronIndexB = tetrahedronB[tid]
    tetrahedronIndexC = tetrahedronC[tid]
    tetrahedronIndexD = tetrahedronD[tid]

    tetrahedronPositionA = predictedVertex[tetrahedronIndexA]
    tetrahedronPositionB = predictedVertex[tetrahedronIndexB]
    tetrahedronPositionC = predictedVertex[tetrahedronIndexC]
    tetrahedronPositionD = predictedVertex[tetrahedronIndexD]

    p1 = tetrahedronPositionB - tetrahedronPositionA
    p2 = tetrahedronPositionC - tetrahedronPositionA
    p3 = tetrahedronPositionD - tetrahedronPositionA

    mA = inverseMass[tetrahedronIndexA]
    mB = inverseMass[tetrahedronIndexB]
    mC = inverseMass[tetrahedronIndexC]
    mD = inverseMass[tetrahedronIndexD]

    invRestPositionsMatrix = inverseRestPositions[tid]

    positionsMatrix = wp.mat33(p1, p2, p3)

    F = wp.mul(positionsMatrix, invRestPositionsMatrix)

    dF1 = wp.cross(F[1], F[2])
    dF2 = wp.cross(F[2], F[0])
    dF3 = wp.cross(F[0], F[1])

    dF = wp.mat33(dF1, dF2, dF3)
    invRestPosMatrixTranspose = wp.transpose(invRestPositionsMatrix)

    g = wp.mul(dF, invRestPosMatrixTranspose)
    # Additional transpose - bug in WARP?
    g = wp.transpose(g)

    g0 = -g[0] - g[1] - g[2]

    volume = wp.determinant(F)
    invVolumeCompliance = 1.0 / volumeCompliance
    invDeviatoricCompliance = 1.0 / deviatoricCompliance
    C = volume - 1.0 - invDeviatoricCompliance / invVolumeCompliance

    w = 0.0
    w += wp.length(g0) * wp.length(g0) * mA
    w += wp.length(g[0]) * wp.length(g[0]) * mB
    w += wp.length(g[1]) * wp.length(g[1]) * mC
    w += wp.length(g[2]) * wp.length(g[2]) * mD
    
    if w < FLOAT_EPSILON:
        return

    alpha = volumeCompliance / dT / dT / restVolume
    dlambda = -C / (w + alpha)

    lmbd += dlambda

    deltaA = g0 * dlambda * mA
    deltaB = g[0] * dlambda * mB
    deltaC = g[1] * dlambda * mC
    deltaD = g[2] * dlambda * mD

    r_s = wp.sqrt(wp.length(F[0]) * wp.length(F[0]) + wp.length(F[1]) * wp.length(F[1]) + wp.length(F[2]) * wp.length(F[2]))
    r_s_inv = 1.0 / r_s

    C = r_s

    g = math.mul(F, invRestPosMatrixTranspose) * r_s_inv
    g = wp.transpose(g)

    g0 = -g[0] - g[1] - g[2]

    w = 0.0
    w += wp.length(g0) * wp.length(g0) * mA
    w += wp.length(g[0]) * wp.length(g[0]) * mB
    w += wp.length(g[1]) * wp.length(g[1]) * mC
    w += wp.length(g[2]) * wp.length(g[2]) * mD
    
    if w < FLOAT_EPSILON:
        return

    alpha = deviatoricCompliance / dT / dT / restVolume
    dlambda = -C / (w + alpha)

    lmbd += dlambda

    deltaA += g0 * dlambda * mA
    deltaB += g[0] * dlambda * mB
    deltaC += g[1] * dlambda * mC
    deltaD += g[2] * dlambda * mD
            
    wp.atomic_add(dP, tetrahedronIndexA, deltaA)
    wp.atomic_add(dP, tetrahedronIndexB, deltaB)
    wp.atomic_add(dP, tetrahedronIndexC, deltaC)
    wp.atomic_add(dP, tetrahedronIndexD, deltaD)

    wp.atomic_add(constraintsNumber, tetrahedronIndexA, 1)
    wp.atomic_add(constraintsNumber, tetrahedronIndexB, 1)
    wp.atomic_add(constraintsNumber, tetrahedronIndexC, 1)
    wp.atomic_add(constraintsNumber, tetrahedronIndexD, 1)

@wp.kernel
def edgeBreakerKernel(predictedVertex: wp.array(dtype=wp.vec3),
                      edge: wp.array(dtype=int),
                      edgeRestLength: wp.array(dtype=float),
                      edgeBreakerValue: wp.array(dtype=float),
                      activeEdge: wp.array(dtype=float),
                      breakFactor: float,
                      breakThreshold: float):
    
    tid = wp.tid()

    if(activeEdge[tid] == 0.0):
        return

    edgeIndexA = edge[tid * 2]
    edgeIndexB = edge[tid * 2 + 1]

    edgePositionA = predictedVertex[edgeIndexA]
    edgePositionB = predictedVertex[edgeIndexB]
    
    edgeRestLen = edgeRestLength[tid]
    if (edgeRestLen < FLOAT_EPSILON):
        return

    dir = edgePositionB - edgePositionA
    len = wp.length(dir)

    if (len < FLOAT_EPSILON):
        return

    C = len/edgeRestLen - 1.0
    if (C > breakThreshold):
        wp.atomic_add(edgeBreakerValue, tid, (C - breakThreshold) * breakFactor)

@wp.kernel
def checkEdgeBreakerKernel(edgeBreakerValue: wp.array(dtype=float),
                           edge: wp.array(dtype=wp.int32),
                           activeEdge: wp.array(dtype=float),
                           flipEdges: wp.array(dtype=wp.int32),
                           numTetrahedrons: wp.int32,
                           tetrahedron: wp.array(dtype=wp.int32),
                           activeTetrahedron: wp.array(dtype=float),
                           flipTetrahedron: wp.array(dtype=wp.int32),
                           edgeToTriangles: wp.array(dtype=wp.int32),
                           holes: wp.array(dtype=wp.int32),
                           breakLimit: float):

    tid = wp.tid()

    if(edgeBreakerValue[tid] > breakLimit):

        if flipEdges[tid] == 0 and activeEdge[tid] == 1.0:
            flipEdges[tid] = -1

        edgeA = edge[tid * 2]
        edgeB = edge[tid * 2 + 1]

        edgeTriA = edgeToTriangles[tid * 2]
        edgeTriB = edgeToTriangles[tid * 2 + 1]

        holes[edgeTriA] = 1
        holes[edgeTriB] = 1

        for i in range(numTetrahedrons):

            tetrahedronA = tetrahedron[i * 4]
            tetrahedronB = tetrahedron[i * 4 + 1]
            tetrahedronC = tetrahedron[i * 4 + 2]
            tetrahedronD = tetrahedron[i * 4 + 3]

            if (edgeA == tetrahedronA or edgeA == tetrahedronB or edgeA == tetrahedronC or edgeA == tetrahedronD) and\
               (edgeB == tetrahedronA or edgeB == tetrahedronB or edgeB == tetrahedronC or edgeB == tetrahedronD):

                if flipTetrahedron[tid] == 0 and activeTetrahedron[tid] == 1.0:
                    flipTetrahedron[tid] = -1

@wp.kernel
def tetrahedronBreaker(predictedVertex: wp.array(dtype=wp.vec3),
                       tetrahedronA: wp.array(dtype=int),
                       tetrahedronB: wp.array(dtype=int),
                       tetrahedronC: wp.array(dtype=int),
                       tetrahedronD: wp.array(dtype=int),
                       tetrahedronRestVolume: wp.array(dtype=float),
                       activeTetrahedron: wp.array(dtype=float),
                       tetrahedronBreakerValue: wp.array(dtype=float),
                       breakFactor: float):
    
    tid = wp.tid()

    if not activeTetrahedron[tid] == 1.0:
        return

    tetrahedronIndexA = tetrahedronA[tid]
    tetrahedronIndexB = tetrahedronB[tid]
    tetrahedronIndexC = tetrahedronC[tid]
    tetrahedronIndexD = tetrahedronD[tid]

    tetrahedronPositionA = predictedVertex[tetrahedronIndexA]
    tetrahedronPositionB = predictedVertex[tetrahedronIndexB]
    tetrahedronPositionC = predictedVertex[tetrahedronIndexC]
    tetrahedronPositionD = predictedVertex[tetrahedronIndexD]
    
    tetrahedronRestVol = tetrahedronRestVolume[tid]

    p1 = tetrahedronPositionB - tetrahedronPositionA
    p2 = tetrahedronPositionC - tetrahedronPositionA
    p3 = tetrahedronPositionD - tetrahedronPositionA

    q2 = wp.cross(p3, p1)
    q1 = wp.cross(p2, p3)
    q3 = wp.cross(p1, p2)
    q0 = - q1 - q2 - q3

    volume = wp.dot(wp.cross(p1, p2), p3) / 6.0

    C = volume - tetrahedronRestVol

    if (C > 0):
        wp.atomic_add(tetrahedronBreakerValue, tid, C*breakFactor)

@wp.kernel
def tetrahedronHeatPropagation(activeTetrahedron: wp.array(dtype=float),
                               tetrahedronNeighbors: wp.array(dtype=int),
                               tetrahedronHeatValue: wp.array(dtype=float),
                               tetrahedronNewHeatValue: wp.array(dtype=float),
                               heatConduction: float,
                               heatTransfer: float,
                               heatLoss: float):
    
    tid = wp.tid()

    if activeTetrahedron[tid] == 0.0:
        return

    tetrahedronHeat = tetrahedronHeatValue[tid]
    neighborAverageHeat = 0.0

    for i in range(4):

        tetrahedronNeighbor = tetrahedronNeighbors[tid * 4 + i]

        if tetrahedronNeighbor > -1:
            if activeTetrahedron[tetrahedronNeighbor] > 0.0:

                tetrahedronNeighborHeat = tetrahedronHeatValue[tetrahedronNeighbor]
                neighborAverageHeat += (tetrahedronNeighborHeat - tetrahedronHeat) * heatConduction

    tetrahedronNewHeatValue[tid] = tetrahedronHeat * (1.0 - heatLoss) + neighborAverageHeat * heatTransfer

@wp.kernel
def tetrahedronHeatUpdate(tetrahedronHeatValue: wp.array(dtype=float),
                          tetrahedronNewHeatValue: wp.array(dtype=float),
                          activeTetrahedron: wp.array(dtype=float),
                          tetrahedron: wp.array(dtype=int),
                          vertexHeatValue: wp.array(dtype=float)):
    
    tid = wp.tid()

    if activeTetrahedron[tid] > 0.0:
        tetrahedronHeat = tetrahedronNewHeatValue[tid]

        tetA = tetrahedron[tid * 4]
        tetB = tetrahedron[tid * 4 + 1]
        tetC = tetrahedron[tid * 4 + 2]
        tetD = tetrahedron[tid * 4 + 3]

        tetrahedronHeatValue[tid] = tetrahedronHeat
        vertexHeatValue[tetA] = tetrahedronHeat / 4.0
        vertexHeatValue[tetB] = tetrahedronHeat / 4.0
        vertexHeatValue[tetC] = tetrahedronHeat / 4.0
        vertexHeatValue[tetD] = tetrahedronHeat / 4.0
    else:
        tetrahedronHeatValue[tid] = 0.0

@wp.kernel
def checkTetrahedronHeat(tetrahedronHeatValue: wp.array(dtype=float),
                         flipTetrahedron: wp.array(dtype=int),
                         tetrahedronToTriangle: wp.array(dtype=wp.int32),
                         holes: wp.array(dtype=wp.int32),
                         heatLimit: float):

    tid = wp.tid()

    tetrahedronHeat = tetrahedronHeatValue[tid]

    if(tetrahedronHeat > heatLimit):

        if flipTetrahedron[tid] == 0:
            flipTetrahedron[tid] = -1

            for i in range(4):
                triangleId = tetrahedronToTriangle[tid * 4 + i]

                if triangleId > -1:
                    # This works for now as we only have external surface triangles
                    holes[triangleId] = 1

@wp.kernel
def checkTetrahedronHeatVolumeProportional(tetrahedronHeatValue: wp.array(dtype=float),
                                           flipTetrahedron: wp.array(dtype=int),
                                           tetrahedronRestVolume: wp.array(dtype=float),
                                           tetrahedronToTriangle: wp.array(dtype=wp.int32),
                                           holes: wp.array(dtype=wp.int32),
                                           heatLimit: float):

    tid = wp.tid()

    tetrahedronHeat = tetrahedronHeatValue[tid]
    tetrahedronRestVol = tetrahedronRestVolume[tid]

    if(tetrahedronHeat > heatLimit * tetrahedronRestVol):

        flipTetrahedron[tid] = -1

        for i in range(4):
            triangleId = tetrahedronToTriangle[tid * 4 + i]

            if triangleId > -1:
                # This works for now as we only have external surface triangles
                holes[triangleId] = 1

@wp.kernel
def zeroDragArray(flipArray: wp.array(dtype=int),
                  activeArray: wp.array(dtype=float)):
    
    tid = wp.tid()
    flip = flipArray[tid]
    active = activeArray[tid]

    if flip > 0:
        flipArray[tid] = -flip
    elif flip == 0 and active > 0.0:
        flipArray[tid] = -1

@wp.kernel
def zeroFloatArray(array: wp.array(dtype=float)):
    
    tid = wp.tid()
    array[tid] = 0.0

@wp.kernel
def zeroVec3Array(array: wp.array(dtype=wp.vec3)):
    
    tid = wp.tid()
    array[tid] = wp.vec3()

@wp.kernel
def applyConstraints(predictedVertex: wp.array(dtype=wp.vec3),
                     dP: wp.array(dtype=wp.vec3),
                     constraintsNumber: wp.array(dtype=int)):
    
    tid = wp.tid()

    if(constraintsNumber[tid] > 0):
        tmpDP = dP[tid]
        N = float(constraintsNumber[tid])
        DP = wp.vec3(tmpDP[0]/N, tmpDP[1]/N, tmpDP[2]/N)
        predictedVertex[tid] = predictedVertex[tid] + DP

    dP[tid] = wp.vec3(0.0, 0.0, 0.0)
    constraintsNumber[tid] = 0

@wp.kernel
def sdfSpherePoint(predictedVertex: wp.array(dtype=wp.vec3),
                   sphereCenter: wp.vec3,
                   sphereRadius: float,
                   kSContact: float):

    tid = wp.tid()

    predictedPosition = predictedVertex[tid]

    contactVector = predictedPosition-sphereCenter

    dist = wp.length(contactVector)
    norm = wp.normalize(contactVector)

    if(dist < sphereRadius):
        contactDP = predictedVertex[tid] + norm * (sphereRadius - dist) * kSContact
        predictedVertex[tid] = contactDP

@wp.kernel
def collideLaparoscopeDO(predictedVertex: wp.array(dtype=wp.vec3),
                         triIds: wp.array(dtype = int),
                         triToEnv: wp.array(dtype=wp.int32),
                         holes: wp.array(dtype=wp.int32),
                         inverseMass: wp.array(dtype=wp.float32),
                         capsuleBases: wp.array(dtype = wp.vec3),
                         capsuleTips: wp.array(dtype = wp.vec3),
                         capsuleRadii: wp.array(dtype = float),
                         inCollision: wp.array(dtype = wp.int32),
                         kSContact: float):
                       
    tid = wp.tid()
    numEnv = triToEnv[tid]

    if (holes[tid] == 1):
        return

    triA = triIds[tid * 3]
    triB = triIds[tid * 3 + 1]
    triC = triIds[tid * 3 + 2]

    pA = predictedVertex[triA]
    pB = predictedVertex[triB]
    pC = predictedVertex[triC]

    triCentre = (pA + pB + pC) * 0.3333
    triNormal = wp.normalize(wp.cross(pB - pA, pC - pA))

    totalResponseA = wp.vec3(0.0, 0.0, 0.0)
    totalResponseB = wp.vec3(0.0, 0.0, 0.0)
    totalResponseC = wp.vec3(0.0, 0.0, 0.0)

    totalResponseLength = 0.0

    for i in range(3):
        capsuleBase = capsuleBases[numEnv * 4 + i]
        capsuleTip = capsuleTips[numEnv * 4 + i]
        capsuleRadius = capsuleRadii[numEnv * 4 + i]

        response = checkSdfCapsuleCollision(capsuleTip,
                                            capsuleBase,
                                            capsuleRadius,
                                            triNormal,
                                            triCentre,
                                            pA,
                                            pB,
                                            pC,
                                            kSContact,
                                            tid)

        responseA = wp.vec3(response[0][0], response[1][0], response[2][0])
        responseB = wp.vec3(response[0][1], response[1][1], response[2][1])
        responseC = wp.vec3(response[0][2], response[1][2], response[2][2])

        totalResponseA = totalResponseA + responseA
        totalResponseB = totalResponseB + responseB
        totalResponseC = totalResponseC + responseC

        totalResponseLength = totalResponseLength + wp.length(responseA) + wp.length(responseB) + wp.length(responseC)

    if totalResponseLength > 0.0:
        wp.atomic_add(inCollision, numEnv, 1)
    
    if(inverseMass[triA] > 0.0):
        wp.atomic_sub(predictedVertex, triA, totalResponseA)
    if(inverseMass[triB] > 0.0):
        wp.atomic_sub(predictedVertex, triB, totalResponseB)
    if(inverseMass[triC] > 0.0):
        wp.atomic_sub(predictedVertex, triC, totalResponseC)

@wp.kernel
def collideLaparoscope(predictedVertex: wp.array(dtype=wp.vec3),
                       triIds: wp.array(dtype = int),
                       holes: wp.array(dtype=wp.int32),
                       inverseMass: wp.array(dtype=wp.float32),
                       capsuleBases: wp.array(dtype = wp.vec3),
                       capsuleTips: wp.array(dtype = wp.vec3),
                       capsuleRadii: wp.array(dtype = float),
                       kSContact: float):
                       
    tid = wp.tid()

    if (holes[tid] == 1):
        return

    triA = triIds[tid * 3]
    triB = triIds[tid * 3 + 1]
    triC = triIds[tid * 3 + 2]

    pA = predictedVertex[triA]
    pB = predictedVertex[triB]
    pC = predictedVertex[triC]

    triCentre = (pA + pB + pC) * 0.3333
    triNormal = wp.normalize(wp.cross(pB - pA, pC - pA))

    totalResponseA = wp.vec3(0.0, 0.0, 0.0)
    totalResponseB = wp.vec3(0.0, 0.0, 0.0)
    totalResponseC = wp.vec3(0.0, 0.0, 0.0)

    for i in range(3):
        capsuleBase = capsuleBases[i]
        capsuleTip = capsuleTips[i]
        capsuleRadius = capsuleRadii[i]

        response = checkSdfCapsuleCollision(capsuleTip,
                                            capsuleBase,
                                            capsuleRadius,
                                            triNormal,
                                            triCentre,
                                            pA,
                                            pB,
                                            pC,
                                            kSContact,
                                            tid)

        responseA = wp.vec3(response[0][0], response[1][0], response[2][0])
        responseB = wp.vec3(response[0][1], response[1][1], response[2][1])
        responseC = wp.vec3(response[0][2], response[1][2], response[2][2])

        totalResponseA = totalResponseA + responseA
        totalResponseB = totalResponseB + responseB
        totalResponseC = totalResponseC + responseC
    
    if(inverseMass[triA] > 0.0):
        wp.atomic_sub(predictedVertex, triA, totalResponseA)
    if(inverseMass[triB] > 0.0):
        wp.atomic_sub(predictedVertex, triB, totalResponseB)
    if(inverseMass[triC] > 0.0):
        wp.atomic_sub(predictedVertex, triC, totalResponseC)

@wp.kernel
def collideElasticRod(triPredictedVertex: wp.array(dtype=wp.vec3),
                      triIds: wp.array(dtype = int),
                      triHoles: wp.array(dtype=wp.int32),
                      triInverseMass: wp.array(dtype=wp.float32),
                      rodPredictedVertex: wp.array(dtype=wp.vec3),
                      rodInverseMass: wp.array(dtype=wp.float32),
                      rodElementRadius: float,
                      kSContact: float):
                       
    triId, elementId = wp.tid()

    if (triHoles[triId] == 1):
        return

    triA = triIds[triId * 3]
    triB = triIds[triId * 3 + 1]
    triC = triIds[triId * 3 + 2]

    pA = triPredictedVertex[triA]
    pB = triPredictedVertex[triB]
    pC = triPredictedVertex[triC]

    pRod = rodPredictedVertex[elementId]

    bary = triangle_closest_point_barycentric(pA, pB, pC, pRod)
    closest = pA * bary[0] + pB * bary[1] + pC * bary[2]

    contactVector = closest-pRod

    dist = wp.length(contactVector)
    norm = wp.normalize(contactVector)

    if(dist < rodElementRadius):
        contactDP = norm * (rodElementRadius - dist) * kSContact

        if(rodInverseMass[elementId] > 0.0):
            wp.atomic_sub(rodPredictedVertex, elementId, contactDP)

@wp.kernel
def heatLaparoscope(triIds: wp.array(dtype=int),
                    vertex: wp.array(dtype=wp.vec3),
                    activeTetrahedron: wp.array(dtype=float),
                    tetrahedronHeatValue: wp.array(dtype=float),
                    triangleToTetrahedron: wp.array(dtype=wp.int32),
                    capsuleTips: wp.array(dtype = wp.vec3),
                    sphereRadius: float,
                    heatQuantum: float):

    tid = wp.tid()

    sphereCenter = (capsuleTips[1] + capsuleTips[2]) / 2.0

    triA = triIds[tid * 3]
    triB = triIds[tid * 3 + 1]
    triC = triIds[tid * 3 + 2]

    pA = vertex[triA]
    pB = vertex[triB]
    pC = vertex[triC]

    bary = triangle_closest_point_barycentric(pA, pB, pC, sphereCenter)
    closest = pA * bary[0] + pB * bary[1] + pC * bary[2]

    contactVector = closest-sphereCenter
    dist = wp.length(contactVector)
    
    if(dist < sphereRadius):
        tetrahedronId = triangleToTetrahedron[tid]
        tetrahedronHeat = tetrahedronHeatValue[tetrahedronId]

        if activeTetrahedron[tetrahedronId] > 0.0:
            tetrahedronHeatValue[tetrahedronId] = tetrahedronHeat + heatQuantum

@wp.kernel
def collideLaparoscopeDebug(predictedVertex: wp.array(dtype=wp.vec3),
                       triIds: wp.array(dtype = int),
                       holes: wp.array(dtype=wp.int32),
                       inverseMass: wp.array(dtype=wp.float32),
                       capsuleBases: wp.array(dtype = wp.vec3),
                       capsuleTips: wp.array(dtype = wp.vec3),
                       capsuleRadii: wp.array(dtype = float),
                       kSContact: float,
                       contactVector: wp.array(dtype = wp.vec3),
                       contactTriangle: wp.array(dtype = wp.int32)):
    
    tid = wp.tid()

    if (holes[tid] == 1):
        return

    triA = triIds[tid * 3]
    triB = triIds[tid * 3 + 1]
    triC = triIds[tid * 3 + 2]

    pA = predictedVertex[triA]
    pB = predictedVertex[triB]
    pC = predictedVertex[triC]

    triCentre = (pA + pB + pC) * 0.3333
    triNormal = wp.normalize(wp.cross(pB - pA, pC - pA))

    totalResponseA = wp.vec3(0.0, 0.0, 0.0)
    totalResponseB = wp.vec3(0.0, 0.0, 0.0)
    totalResponseC = wp.vec3(0.0, 0.0, 0.0)

    for i in range(3):
        capsuleBase = capsuleBases[i]
        capsuleTip = capsuleTips[i]
        capsuleRadius = capsuleRadii[i]

        response = checkSdfCapsuleCollision(capsuleTip,
                                            capsuleBase,
                                            capsuleRadius,
                                            triNormal,
                                            triCentre,
                                            pA,
                                            pB,
                                            pC,
                                            kSContact,
                                            tid)

        responseA = wp.vec3(response[0][0], response[1][0], response[2][0])
        responseB = wp.vec3(response[0][1], response[1][1], response[2][1])
        responseC = wp.vec3(response[0][2], response[1][2], response[2][2])

        totalResponseA = totalResponseA + responseA
        totalResponseB = totalResponseB + responseB
        totalResponseC = totalResponseC + responseC

    if(wp.length(responseA + responseB + responseC) > 0.0):
        contactTriangle[tid] = 1
    else:
        contactTriangle[tid] = 0

    if(inverseMass[triA] > 0.0):
        wp.atomic_add(contactVector, triA, totalResponseA)
    if(inverseMass[triB] > 0.0):
        wp.atomic_add(contactVector, triB, totalResponseB)
    if(inverseMass[triC] > 0.0):
        wp.atomic_add(contactVector, triC, totalResponseC)

    if(inverseMass[triA] > 0.0):
        wp.atomic_sub(predictedVertex, triA, totalResponseA)
    if(inverseMass[triB] > 0.0):
        wp.atomic_sub(predictedVertex, triB, totalResponseB)
    if(inverseMass[triC] > 0.0):
        wp.atomic_sub(predictedVertex, triC, totalResponseC)
    
@wp.kernel
def sdfCapsuleTriangle(predictedVertex: wp.array(dtype=wp.vec3),
              triIds: wp.array(dtype = int),
              capsuleBase: wp.vec3,
              capsuleTip: wp.vec3,
              capsuleRadius: float,
              kSContact: float):

    tid = wp.tid()

    triA = triIds[tid * 3]
    triB = triIds[tid * 3 + 1]
    triC = triIds[tid * 3 + 2]

    pA = predictedVertex[triA]
    pB = predictedVertex[triB]
    pC = predictedVertex[triC]

    triCentre = (pA + pB + pC) * 0.3333
    triNormal = wp.normalize(wp.cross(pB - pA, pC - pA))

    response = checkSdfCapsuleCollision(capsuleTip,
                                        capsuleBase,
                                        capsuleRadius,
                                        triNormal,
                                        triCentre,
                                        pA,
                                        pB,
                                        pC,
                                        kSContact,
                                        tid)

    responseA = wp.vec3(response[0][0], response[1][0], response[2][0])
    responseB = wp.vec3(response[0][1], response[1][1], response[2][1])
    responseC = wp.vec3(response[0][2], response[1][2], response[2][2])

    predictedVertex[triA] = predictedVertex[triA] - responseA
    predictedVertex[triB] = predictedVertex[triB] - responseB
    predictedVertex[triC] = predictedVertex[triC] - responseC

@wp.func
def checkSdfCapsuleCollision(capsuleTip: wp.vec3,
                             capsuleBase: wp.vec3,
                             capsuleRadius: float,
                             triNormal: wp.vec3,
                             triCentre: wp.vec3,
                             pA: wp.vec3,
                             pB: wp.vec3,
                             pC: wp.vec3,
                             kSContact: float,
                             tid: wp.int32) -> wp.mat33:

    responseMat = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # ToDo: This can be calculated on cpu once per frame for each capsule
    capsuleNormal = wp.normalize(capsuleTip - capsuleBase)
    lineEndOffset = capsuleNormal * capsuleRadius

    posA = capsuleBase + lineEndOffset
    posB = capsuleTip - lineEndOffset

    denom = wp.dot(triNormal, capsuleNormal)

    if(wp.abs(denom) < FLOAT_EPSILON):
        return responseMat

    t = wp.dot(triNormal, (triCentre - capsuleBase)) / denom

    linePlaneIntersection = capsuleBase + capsuleNormal * t

    referencePoint = triangle_closest_point_barycentric(pA, pB, pC, linePlaneIntersection)
    sphereCenter = closestPointOnEdge(referencePoint, posA, posB)

    bary = triangle_closest_point_barycentric(pA, pB, pC, sphereCenter)
    contactPoint = pA * bary[0] + pB * bary[1] + pC * bary[2]

    sphereCenterToContactPoint = sphereCenter - contactPoint
    distance = wp.length(sphereCenterToContactPoint)
    penetration = (capsuleRadius / wp.length(sphereCenterToContactPoint) - 1.0)
    penetrationVector = wp.vec3(penetration * sphereCenterToContactPoint[0], penetration * sphereCenterToContactPoint[1], penetration * sphereCenterToContactPoint[2])
    penetrationLength = wp.length(penetrationVector)
    penetrationNormal = wp.normalize(penetrationVector)

    s = bary[0] * bary[0] + bary[1] * bary[1] + bary[2] * bary[2]
    if (s < FLOAT_EPSILON):
        return responseMat

    response = penetrationNormal * (penetrationLength / s) * kSContact
    
    if distance > capsuleRadius:
        return responseMat

    responseMat = wp.mat33(response * bary[0], response * bary[1], response * bary[2])
    return responseMat

@wp.kernel
def sdfSphereTriangle(predictedVertex: wp.array(dtype=wp.vec3),
                      triIds: wp.array(dtype=int),
                      sphereCenter: wp.vec3,
                      sphereRadius: float,
                      kSContact: float):

    tid = wp.tid()

    triA = triIds[tid * 3]
    triB = triIds[tid * 3 + 1]
    triC = triIds[tid * 3 + 2]

    pA = predictedVertex[triA]
    pB = predictedVertex[triB]
    pC = predictedVertex[triC]

    bary = triangle_closest_point_barycentric(pA, pB, pC, sphereCenter)
    closest = pA * bary[0] + pB * bary[1] + pC * bary[2]

    contactVector = closest-sphereCenter
    dist = wp.length(contactVector)
    norm = wp.normalize(contactVector)

    if(dist < sphereRadius):
        contactDP = norm * (sphereRadius - dist) * kSContact
        predictedVertex[triA] = predictedVertex[triA] + contactDP * bary[0]
        predictedVertex[triB] = predictedVertex[triB] + contactDP * bary[1]
        predictedVertex[triC] = predictedVertex[triC] + contactDP * bary[2]

@wp.kernel
def findClosestPoint(vertex: wp.array(dtype=wp.vec3),
                     laparoscopeInfo: wp.array(dtype = wp.vec3),
                     flipConstraints: wp.array(dtype = wp.int32),
                     lookupRadius: float):

    tid = wp.tid()

    position = vertex[tid]
    laparoscopeDragPoint = laparoscopeInfo[3]

    length = wp.length(position - laparoscopeDragPoint)

    if (length < lookupRadius):
        flipConstraints[tid] = 1
    else:
        flipConstraints[tid] = 0

@wp.kernel
def findIntersectingEdges(vertex: wp.array(dtype=wp.vec3),
                          laparoscopeInfo: wp.array(dtype = wp.vec3),
                          edge: wp.array(dtype=wp.int32),
                          activeEdge: wp.array(dtype=float),
                          flipEdges: wp.array(dtype=wp.int32),
                          numTetrahedrons: wp.int32,
                          tetrahedron: wp.array(dtype=wp.int32),
                          activeTetrahedron: wp.array(dtype=float),
                          flipTetrahedron: wp.array(dtype=wp.int32),
                          edgeToTriangles: wp.array(dtype=wp.int32),
                          holes: wp.array(dtype=wp.int32)):

    tid = wp.tid()

    laparoscopeRodTip = laparoscopeInfo[0]
    laparoscopeLeftClampTip = laparoscopeInfo[1]
    laparoscopeRightClampTip = laparoscopeInfo[2]

    edgeA = edge[tid * 2]
    edgeB = edge[tid * 2 + 1]

    sA = vertex[edgeA]
    sB = vertex[edgeB]

    intersection = checkSegmentTriangleIntersection(laparoscopeRodTip,
                                                    laparoscopeRightClampTip,
                                                    laparoscopeLeftClampTip,
                                                    sA,
                                                    sB)

    if(intersection == 1):
        if flipEdges[tid] == 0 and activeEdge[tid] == 1.0:
            flipEdges[tid] = -1

        edgeTriA = edgeToTriangles[tid * 2]
        edgeTriB = edgeToTriangles[tid * 2 + 1]

        holes[edgeTriA] = 1
        holes[edgeTriB] = 1

        for i in range(numTetrahedrons):

            tetrahedronA = tetrahedron[i * 4]
            tetrahedronB = tetrahedron[i * 4 + 1]
            tetrahedronC = tetrahedron[i * 4 + 2]
            tetrahedronD = tetrahedron[i * 4 + 3]

            if (edgeA == tetrahedronA or edgeA == tetrahedronB or edgeA == tetrahedronC or edgeA == tetrahedronD) and\
               (edgeB == tetrahedronA or edgeB == tetrahedronB or edgeB == tetrahedronC or edgeB == tetrahedronD):

                if flipTetrahedron[i] == 0 and activeTetrahedron[i] == 1.0:
                    flipTetrahedron[i] = -1

@wp.kernel
def remapTrianglesToFaces(triangles: wp.array(dtype=wp.int32),
                          faces: wp.array(dtype=wp.int32),
                          triangleToFace: wp.array(dtype=wp.int32)):

    tid = wp.tid()

    triangleId = triangles[tid]
    faceId = triangleToFace[triangleId]

    faces[tid] = faceId

@wp.func
def checkSegmentTriangleIntersection(pA: wp.vec3, pB: wp.vec3, pC:wp.vec3, sA: wp.vec3, sB: wp.vec3) -> int:
    u = pB - pA
    v = pC - pA
    n = wp.cross(u, v)

    if (n == wp.vec3()):
        return -1

    dir = sB - sA
    w0 = sA - pA
    a = -wp.dot(n, w0)
    b = wp.dot(n, dir)
    if (wp.abs(b) < FLOAT_EPSILON):
        if (a == 0.0):
            return 2
        else:
            return 0

    r = a/b
    if(r < 0.0):
        return 0
    if(r > 1.0):
        return 0

    I = sA + r * dir

    uu = wp.dot(u, u)
    uv = wp.dot(u, v)
    vv = wp.dot(v, v)

    w = I - pA
    wu = wp.dot(w, u)
    wv = wp.dot(w, v)
    D = uv * uv - uu * vv

    s = (uv * wv - vv * wu) / D
    if (s < 0.0):
        return 0
    if (s > 1.0):
        return 0
    t = (uv * wu - uu * wv) / D
    if(t < 0.0):
        return 0
    if((s+t) > 1.0):
        return 0

    return 1

@wp.func
def closestPointOnEdge(p: wp.vec3, edgeA: wp.vec3, edgeB: wp.vec3):
    d = edgeB - edgeA

    if wp.length(d) < FLOAT_EPSILON:
        t = 0.5
    else:
        d2 = wp.length(d) * wp.length(d)
        t = wp.dot(d, p-edgeA) / d2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0

    return edgeA + d * t

@wp.func
def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)

    if (d1 <= 0.0 and d2 <= 0.0):
        return wp.vec3(1.0, 0.0, 0.0)

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)

    if (d3 >= 0.0 and d4 <= d3):
        return wp.vec3(0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if (vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0):
        return wp.vec3(1.0 - v, v, 0.0)

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)

    if (d6 >= 0.0 and d5 <= d6):
        return wp.vec3(0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if (vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0):
        return wp.vec3(1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if (va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0):
        return wp.vec3(0.0, w, 1.0 - w)

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom

    return wp.vec3(1.0 - v - w, v, w)

@wp.kernel
def inDeCreaseActiveValues(flipArray: wp.array(dtype=int),
                           activeArray: wp.array(dtype=float),
                           sqrtValues: wp.array(dtype=float)):

    tid = wp.tid()

    flip = flipArray[tid]

    if not flip == 0:
        activeStep = 0
        if flip > 0:
            activeStep = flip
            flipArray[tid] = flip + 1
        else:
            activeStep = IN_DE_CREASE_STEPS + flip
            flipArray[tid] = flip - 1
        
        sqrtValue  = sqrtValues[activeStep]
        activeArray[tid] = sqrtValue

        if(wp.abs(flip) >= IN_DE_CREASE_STEPS):
            flipArray[tid] = 0

@wp.kernel
def moveElasticRod(elementVertex: wp.array(dtype=wp.vec3),
                   currentPosition: wp.vec3):

    tid = wp.tid()

    if tid == 0:
        elementVertex[tid] = currentPosition

@wp.kernel
def moveVelocity(vertex: wp.array(dtype=wp.vec3),
                 velocity: wp.array(dtype=wp.vec3),
                 gravityConstant: wp.vec3,
                 velocityDampening: float,
                 dt: wp.float32):
    
    tid = wp.tid()

    vert = vertex[tid]
    velo = velocity[tid]

    velo += gravityConstant * dt
    vert += velo * dt
    
    vertex[tid] = vert
    velocity[tid] = velo

class SimIntegratorDO():

    def __init__(self, device="cuda:0") -> None:
        self.device = device
        print("Integrator initialized")

    def stepModel(self, simModel:dk.SimModel) -> None:

        for i in range(simModel.simSubsteps):

            wp.launch(kernel=gravity,
                      dim=simModel.numVertices,
                      inputs=[simModel.vertex,
                              simModel.predictedVertex,
                              simModel.velocity,
                              simModel.inverseMass,
                              simModel.gravity,
                              simModel.velocityDampening,
                              simModel.simDt],
                      device=simModel.device)

            # Evaluate constraints
            for j in range(simModel.simConstraints):
            
                wp.launch(kernel=volumeConstraints,
                          dim=simModel.numTetrahedrons,
                          inputs=[simModel.predictedVertex,
                                  simModel.dP,
                                  simModel.constraintsNumber,
                                  simModel.tetrahedron,
                                  simModel.tetrahedronRestVolume,
                                  simModel.inverseMass,
                                  simModel.activeTetrahedron,
                                  simModel.globalKsVolume],
                          device=simModel.device)

                wp.launch(kernel=distanceConstraints,
                          dim=simModel.numEdges,
                          inputs=[simModel.predictedVertex,
                                  simModel.dP,
                                  simModel.constraintsNumber,
                                  simModel.edge,
                                  simModel.edgeRestLength,
                                  simModel.inverseMass,
                                  simModel.activeEdge,
                                  simModel.globalKsDistance],
                          device=simModel.device)
                
                wp.launch(kernel=applyConstraints,
                          dim=simModel.numVertices,
                          inputs=[simModel.predictedVertex,
                                  simModel.dP,
                                  simModel.constraintsNumber],
                          device=simModel.device)
                
                wp.launch(kernel=dragConstraintsDO,
                          dim=len(simModel.predictedVertex),
                          inputs=[simModel.predictedVertex,
                                  simModel.activeDragConstraint,
                                  simModel.vertexToEnv,
                                  simModel.laparoscopeTip,
                                  simModel.globalKsDrag],
                          device=simModel.device)
                
                wp.launch(
                    kernel=triToPointDistanceConstraintsDO,
                    dim=simModel.simConnectors,
                    inputs=[simModel.predictedVertex,
                            simModel.dP,
                            simModel.constraintsNumber,
                            simModel.connectorTriangleId,
                            simModel.connectorTriBar,
                            simModel.connectorVertexId,
                            simModel.connectorRestDist,
                            simModel.globalConnectorMassWeightRatio,
                            simModel.globalConnectorKs,
                            simModel.globalConnectorRestLengthMul],
                    device=simModel.device)
                
                wp.launch(
                    kernel=collideLaparoscopeDO,
                    dim=simModel.numTriangles,
                    inputs=[simModel.predictedVertex,
                            simModel.triangle,
                            simModel.triToEnv,
                            simModel.hole,
                            simModel.inverseMass,
                            simModel.laparoscopeBase,
                            simModel.laparoscopeTip,
                            simModel.laparoscopeRadius,
                            simModel.laparoscopeInCollision,
                            simModel.globalKsContact],
                    device=simModel.device)
                
            # Check collision with floor
            wp.launch(kernel=bounds,
                      dim=len(simModel.vertex),
                      inputs=[simModel.predictedVertex,
                              simModel.groundLevel],
                      device=simModel.device)
            
            wp.launch(kernel=PBDStep,
                      dim=len(simModel.vertex),
                      inputs=[simModel.vertex,
                              simModel.predictedVertex,
                              simModel.velocity,
                              simModel.simDt],
                      device=simModel.device) 
            
        simModel.transformSimMeshData()
        simModel.transformSimLaparoscopeMeshData()

class SimIntegrator():
    def __init__(self, device) -> None:
        self.device = device
        print("Integrator initialized")
        
    def testDiff(self, simModel:dk.SimModel) -> None:

        for simEnvironment in simModel.simEnvironments:
            for i in range(simModel.simSubsteps):
                # Run gravity on all objects
                for simMesh in simEnvironment.simMeshes:
                    wp.launch(kernel=gravity,
                        dim=len(simMesh.vertex),
                        inputs=[simMesh.vertex,
                                simMesh.predictedVertex,
                                simMesh.velocity,
                                simMesh.inverseMass,
                                simModel.gravity,
                                simModel.velocityDampening,
                                simModel.simDt],
                        device=simModel.device)

                    wp.launch(kernel=PBDStep,
                        dim=len(simMesh.vertex),
                        inputs=[simMesh.vertex,
                                simMesh.predictedVertex,
                                simMesh.velocity,
                                simModel.simDt],
                        device=simModel.device)

                    # # Propagate data to visual mesh
                    # simMesh.remapGlobalPointsFromVerticesWarp()
        
    def stepModel(self, simModel:dk.SimModel) -> None:

        for simEnvironment in simModel.simEnvironments:

            for i in range(simModel.simSubsteps):

                # Run gravity on all objects
                for simMesh in simEnvironment.simMeshes:

                    wp.launch(kernel=gravity,
                        dim=len(simMesh.vertex),
                        inputs=[simMesh.vertex,
                                simMesh.predictedVertex,
                                simMesh.velocity,
                                simMesh.inverseMass,
                                simModel.gravity,
                                simModel.velocityDampening,
                                simModel.simDt],
                        device=simModel.device)

                # Evaluate constraints
                for j in range(simModel.simConstraints):

                    for simMesh in simEnvironment.simMeshes:

                        wp.launch(
                            kernel=volumeConstraints,
                            dim=simMesh.numTetrahedrons,
                            inputs=[simMesh.predictedVertex,
                                    simMesh.dP,
                                    simMesh.constraintsNumber,
                                    simMesh.tetrahedron,
                                    simMesh.tetrahedronRestVolume,
                                    simMesh.inverseMass,
                                    simMesh.activeTetrahedron,
                                    simModel.globalKsVolume * simMesh.ksVolume],
                            device=simModel.device)

                        wp.launch(
                            kernel=distanceConstraints,
                            dim=simMesh.numEdges,
                            inputs=[simMesh.predictedVertex,
                                    simMesh.dP,
                                    simMesh.constraintsNumber,
                                    simMesh.edge,
                                    simMesh.edgeRestLength,
                                    simMesh.inverseMass,
                                    simMesh.activeEdge,
                                    simModel.globalKsDistance * simMesh.ksDistance],
                            device=simModel.device)

                        for simLaparoscope in simEnvironment.simLaparoscopes:
                            # Apply laparoscope drag
                            wp.launch(
                                kernel=dragConstraints,
                                dim=len(simMesh.predictedVertex),
                                inputs=[simMesh.predictedVertex,
                                        simMesh.activeDragConstraint,
                                        simLaparoscope.laparoscopeTip,
                                        simModel.globalKsDrag * simMesh.ksDrag],
                                device=simModel.device)

                        # for simLaparoscope in simEnvironment.simLaparoscopes:
                        #     # Apply laparoscope drag
                        #     wp.launch(
                        #         kernel=dragConstraintsDP,
                        #         dim=len(simMesh.predictedVertex),
                        #         inputs=[simMesh.predictedVertex,
                        #                 simMesh.dP,
                        #                 simMesh.edgeLambda,
                        #                 simMesh.constraintsNumber,
                        #                 simMesh.activeDragConstraint,
                        #                 simLaparoscope.laparoscopeTip,
                        #                 simModel.globalKsDrag * simMesh.ksDrag],
                        #         device=simModel.device)

                    # Evaluate tissue connectors constraints
                    for simConnector in simEnvironment.simConnectors:
                        wp.launch(
                            kernel=triToPointDistanceConstraints,
                            dim=len(simConnector.pointIds),
                            inputs=[simConnector.pointBodyPredictedVertex,
                                    simConnector.triBodyPredictedVertex,
                                    simConnector.pointBodyDP,
                                    simConnector.triBodyDP,
                                    simConnector.pointBodyConstraintsNumber,
                                    simConnector.triBodyConstraintsNumber,
                                    simConnector.triIds,
                                    simConnector.triBar,
                                    simConnector.pointIds,
                                    simConnector.restDist,
                                    simConnector.massWeightRatio,
                                    simConnector.kS,
                                    simConnector.restLengthMul],
                            device=simModel.device)

                    for simMesh in simEnvironment.simMeshes:

                        wp.launch(
                            kernel=applyConstraints,
                            dim=len(simMesh.vertex),
                            inputs=[simMesh.predictedVertex,
                                    simMesh.dP,
                                    simMesh.constraintsNumber],
                            device=simModel.device)

                        # Mesh collisions with laparoscope
                        for simLaparoscope in simEnvironment.simLaparoscopes:
                            wp.launch(
                                kernel=collideLaparoscope,
                                dim=simMesh.numTris,
                                inputs=[simMesh.predictedVertex,
                                        simMesh.triangle,
                                        simMesh.hole,
                                        simMesh.inverseMass,
                                        simLaparoscope.laparoscopeBase,
                                        simLaparoscope.laparoscopeTip,
                                        simLaparoscope.laparoscopeRadius,
                                        simModel.globalKsContact * simMesh.ksContact],
                                device=simModel.device)

                            # # Apply laparoscope drag
                            # wp.launch(
                            #     kernel=dragConstraints,
                            #     dim=len(simMesh.predictedVertex),
                            #     inputs=[simMesh.predictedVertex,
                            #             simMesh.activeDragConstraint,
                            #             simLaparoscope.laparoscopeTip,
                            #             simModel.globalKsDrag * simMesh.ksDrag],
                            #     device=simModel.device)

                for simMesh in simEnvironment.simMeshes:
                    
                    # Check collision with floor
                    wp.launch(kernel=bounds,
                          dim=len(simMesh.predictedVertex),
                          inputs=[simMesh.predictedVertex,
                                  simModel.groundLevel],
                          device=simModel.device)  

                    wp.launch(kernel=PBDStep,
                        dim=len(simMesh.vertex),
                        inputs=[simMesh.vertex,
                                simMesh.predictedVertex,
                                simMesh.velocity,
                                simModel.simDt],
                        device=simModel.device)

            for simMesh in simEnvironment.simMeshes:
                # Propagate data to visual mesh
                simMesh.remapGlobalPointsFromVerticesWarp()
        
    def gravityStep(simModel:dk.SimModel) -> None:
        simContext = simModel.simContext

        for i in range(simContext.simSubsteps):
            # Run gravity on all objects
            for simMesh in simModel.simMeshes:
                wp.launch(kernel=gravity,
                    dim=len(simMesh.vertex),
                    inputs=[simMesh.vertex,
                            simMesh.predictedVertex,
                            simMesh.velocity,
                            simMesh.inverseMass,
                            simContext.gravity,
                            simContext.velocityDampening,
                            simContext.simDt],
                    device=simContext.simDevice)

                wp.launch(kernel=PBDStep,
                    dim=len(simMesh.vertex),
                    inputs=[simMesh.vertex,
                            simMesh.predictedVertex,
                            simMesh.velocity,
                            simContext.simDt],
                    device=simContext.simDevice)
                            
            for simElasticRod in simModel.simElasticRods:
                wp.launch(kernel=gravity,
                    dim=simElasticRod.elementNum,
                    inputs=[simElasticRod.elementVertex,
                            simElasticRod.elementPredictedVertex,
                            simElasticRod.elementVelocity,
                            simElasticRod.elementInverseMass,
                            simContext.gravity,
                            simContext.velocityDampening,
                            simContext.simDt],
                    device=simContext.simDevice)

                wp.launch(kernel=PBDStep,
                    dim=simElasticRod.elementNum,
                    inputs=[simElasticRod.elementVertex,
                            simElasticRod.elementPredictedVertex,
                            simElasticRod.elementVelocity,
                            simContext.simDt],
                    device=simContext.simDevice)
        
    def step(simModel:dk.SimModel) -> None:
        simContext = simModel.simContext

        for i in range(simContext.simSubsteps):
            # Run gravity on all objects
            for simMesh in simModel.simMeshes:
                wp.launch(kernel=gravity,
                    dim=len(simMesh.vertex),
                    inputs=[simMesh.vertex,
                            simMesh.predictedVertex,
                            simMesh.velocity,
                            simMesh.inverseMass,
                            simContext.gravity,
                            simContext.velocityDampening,
                            simContext.simDt],
                    device=simContext.simDevice)
                            
            for simElasticRod in simModel.simElasticRods:
                wp.launch(kernel=gravity,
                    dim=len(simElasticRod.elementVertex),
                    inputs=[simElasticRod.elementVertex,
                            simElasticRod.elementPredictedVertex,
                            simElasticRod.elementVelocity,
                            simElasticRod.elementInverseMass,
                            simContext.gravity,
                            simContext.velocityDampening,
                            simContext.simDt],
                    device=simContext.simDevice)

            # Evaluate constraints
            for j in range(simContext.simConstraints):
                for simMesh in simModel.simMeshes:
                    wp.launch(
                        kernel=volumeConstraints,
                        dim=simMesh.numTetrahedrons,
                        inputs=[simMesh.predictedVertex,
                                simMesh.dP,
                                simMesh.constraintsNumber,
                                simMesh.tetrahedron,
                                simMesh.tetrahedronRestVolume,
                                simMesh.inverseMass,
                                simMesh.activeTetrahedron,
                                simContext.globalKsVolume * simMesh.ksVolume],
                        device=simContext.simDevice)

                    wp.launch(
                        kernel=distanceConstraints,
                        dim=simMesh.numEdges,
                        inputs=[simMesh.predictedVertex,
                                simMesh.dP,
                                simMesh.constraintsNumber,
                                simMesh.edge,
                                simMesh.edgeRestLength,
                                simMesh.inverseMass,
                                simMesh.activeEdge,
                                simContext.globalKsDistance * simMesh.ksDistance],
                        device=simContext.simDevice)

                    # wp.launch(
                    #     kernel=dragConstraints,
                    #     dim=len(simMesh.predictedVertex),
                    #     inputs=[simMesh.predictedVertex,
                    #             simMesh.activeDragConstraint,
                    #             simContext.laparoscope.laparoscopeTips,
                    #             simContext.globalKsDrag * simMesh.ksDrag],
                    #     device=simContext.simDevice)

                # Evaluate tissue connectors constraints
                for simConnector in simModel.simConnectors:
                    wp.launch(
                        kernel=triToPointDistanceConstraints,
                        dim=len(simConnector.pointIds),
                        inputs=[simConnector.pointBodyPredictedVertex,
                                simConnector.triBodyPredictedVertex,
                                simConnector.pointBodyDP,
                                simConnector.triBodyDP,
                                simConnector.pointBodyConstraintsNumber,
                                simConnector.triBodyConstraintsNumber,
                                simConnector.triIds,
                                simConnector.triBar,
                                simConnector.pointIds,
                                simConnector.restDist,
                                simConnector.massWeightRatio,
                                simConnector.kS,
                                simConnector.restLengthMul],
                        device=simContext.simDevice)

                # Evaluate elastic rods constraints
                for simElasticRod in simModel.simElasticRods:
                    if simElasticRod.iterativeRod:
                        wp.launch(
                            kernel=distanceConstraintsIterOdd,
                            dim=math.ceil(simElasticRod.numElementEdge / 2.0),
                            inputs=[simElasticRod.elementPredictedVertex,
                                    simElasticRod.elementEdge,
                                    simElasticRod.elementEdgeRestLength,
                                    simElasticRod.elementInverseMass,
                                    simElasticRod.elementActiveEdge,
                                    simContext.globalKsDistance * simElasticRod.elementKs],
                            device=simContext.simDevice)

                        wp.launch(
                            kernel=distanceConstraintsIterEven,
                            dim=int(simElasticRod.numElementEdge / 2.0),
                            inputs=[simElasticRod.elementPredictedVertex,
                                    simElasticRod.elementEdge,
                                    simElasticRod.elementEdgeRestLength,
                                    simElasticRod.elementInverseMass,
                                    simElasticRod.elementActiveEdge,
                                    simContext.globalKsDistance * simElasticRod.elementKs],
                            device=simContext.simDevice)

                    else:
                        wp.launch(
                            kernel=distanceConstraints,
                            dim=simElasticRod.numElementEdge,
                            inputs=[simElasticRod.elementPredictedVertex,
                                    simElasticRod.elementDp,
                                    simElasticRod.elementEdgeLambda,
                                    simElasticRod.elementConstraintsNumber,
                                    simElasticRod.elementEdge,
                                    simElasticRod.elementEdgeRestLength,
                                    simElasticRod.elementInverseMass,
                                    simElasticRod.elementActiveEdge,
                                    simContext.globalKsDistance * simElasticRod.elementKs],
                            device=simContext.simDevice)
                    
                # Apply constraints for meshes
                for simMesh in simModel.simMeshes:
                    wp.launch(
                        kernel=applyConstraints,
                        dim=len(simMesh.vertex),
                        inputs=[simMesh.predictedVertex,
                                simMesh.dP,
                                simMesh.constraintsNumber],
                        device=simContext.simDevice)

                    # wp.launch(
                    #     kernel=collideLaparoscopeDebug,
                    #     dim=simMesh.numTris,
                    #     inputs=[simMesh.predictedVertex,
                    #             simMesh.triangles,
                    #             simMesh.holes,
                    #             simMesh.inverseMass,
                    #             simContext.laparoscope.laparoscopeBases,
                    #             simContext.laparoscope.laparoscopeTips,
                    #             simContext.laparoscope.laparoscopeRadii,
                    #             simContext.globalKsContact * simMesh.ksContact,
                    #             simMesh.debugContactVector,
                    #             simMesh.debugContactTriangle],
                    #     device=simContext.simDevice)

                # Apply constraints for elastic rods
                for simElasticRod in simModel.simElasticRods:
                    if not simElasticRod.iterativeRod:
                        wp.launch(
                            kernel=applyConstraints,
                            dim=simElasticRod.elementNum,
                            inputs=[simElasticRod.elementPredictedVertex,
                                    simElasticRod.elementDp,
                                    simElasticRod.elementConstraintsNumber],
                            device=simContext.simDevice)

                    for simMesh in simModel.simMeshes:
                        wp.launch(
                            kernel=collideElasticRod,
                            dim=(simMesh.numTris,
                                 simElasticRod.elementNum),
                            inputs=[simMesh.predictedVertex,
                                    simMesh.triangle,
                                    simMesh.hole,
                                    simMesh.inverseMass,
                                    simElasticRod.elementPredictedVertex,
                                    simElasticRod.elementInverseMass,
                                    simElasticRod.elementRadius,
                                    simContext.globalKsContact * simMesh.ksContact],
                            device=simContext.simDevice)

            for simMesh in simModel.simMeshes:
                # Zero lambdas
                if simContext.simConstraints > 1:
                    wp.launch(
                        kernel=zeroFloatArray,
                        dim=len(simMesh.edgeLambda),
                        inputs=[simMesh.edgeLambda],
                        device=simContext.simDevice)

                    wp.launch(
                        kernel=zeroFloatArray,
                        dim=len(simMesh.tetrahedronLambda),
                        inputs=[simMesh.tetrahedronLambda],
                        device=simContext.simDevice)

                    wp.launch(
                        kernel=zeroFloatArray,
                        dim=len(simMesh.neoHookeanLambda),
                        inputs=[simMesh.neoHookeanLambda],
                        device=simContext.simDevice)

                # # Check collision with floor
                # wp.launch(kernel=bounds,
                #       dim=len(simMesh.predictedVertex),
                #       inputs=[simMesh.predictedVertex,
                #               simContext.groundLevel],
                #       device=simContext.simDevice)

                # # Check edge breaking
                # wp.launch(
                #     kernel=edgeBreakerKernel,
                #     dim=simMesh.numEdges,
                #     inputs=[simMesh.predictedVertex,
                #             simMesh.edge,
                #             simMesh.edgeRestLength,
                #             simMesh.edgeBreaker,
                #             simMesh.activeEdge,
                #             simContext.globalBreakFactor * simMesh.breakFactor,
                #             simContext.globalBreakThreshold * simMesh.breakThreshold],
                #     device=simContext.simDevice)

                # wp.launch(kernel=checkEdgeBreakerKernel,
                #     dim=simMesh.numEdges,
                #     inputs=[simMesh.edgeBreaker,
                #             simMesh.edge,
                #             simMesh.activeEdge,
                #             simMesh.flipEdges,
                #             simMesh.numTetrahedrons,
                #             simMesh.tetrahedron,
                #             simMesh.activeTetrahedron,
                #             simMesh.flipTetrahedron,
                #             simMesh.edgeToTriangles,
                #             simMesh.holes,
                #             1.0],
                #     device=simContext.simDevice)

                # # Check heat propagation
                # wp.launch(kernel=tetrahedronHeatPropagation,
                #     dim=simMesh.numTetrahedrons,
                #     inputs=[simMesh.activeTetrahedron,
                #             simMesh.tetrahedronNeighbors,
                #             simMesh.tetrahedronHeat,
                #             simMesh.tetrahedronNewHeat,
                #             simContext.globalHeatConduction * simMesh.heatConduction,
                #             simContext.globalHeatTransfer * simMesh.heatTransfer,
                #             simContext.globalHeatLoss * simMesh.heatLoss],
                #     device=simContext.simDevice)

                # wp.launch(kernel=tetrahedronHeatUpdate,
                #     dim=simMesh.numTetrahedrons,
                #     inputs=[simMesh.tetrahedronHeat,
                #             simMesh.tetrahedronNewHeat,
                #             simMesh.activeTetrahedron,
                #             simMesh.tetrahedron,
                #             simMesh.vertexHeat],
                #     device=simContext.simDevice)

                # Perform the PBD step
                wp.launch(kernel=PBDStep,
                    dim=len(simMesh.vertex),
                    inputs=[simMesh.vertex,
                            simMesh.predictedVertex,
                            simMesh.velocity,
                            simContext.simDt],
                    device=simContext.simDevice)

            for simElasticRod in simModel.simElasticRods:

                # if simContext.simConstraints > 1:
                #     wp.launch(
                #         kernel=zeroFloatArray,
                #         dim=len(simElasticRod.elementEdgeLambda),
                #         inputs=[simElasticRod.elementEdgeLambda],
                #         device=simContext.simDevice)

                # # wp.launch(kernel=bounds,
                # #     dim=len(simElasticRod.elementPredictedVertex),
                # #     inputs=[simElasticRod.elementPredictedVertex,
                # #             simContext.groundLevel],
                # #     device=simContext.simDevice)

                wp.launch(kernel=PBDStep,
                    dim=len(simElasticRod.elementVertex),
                    inputs=[simElasticRod.elementVertex,
                            simElasticRod.elementPredictedVertex,
                            simElasticRod.elementVelocity,
                            simContext.simDt],
                    device=simContext.simDevice)