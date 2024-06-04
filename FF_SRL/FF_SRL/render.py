import warp as wp
import FF_SRL as dk
import numpy as np
import torch
import math
import sys
import matplotlib.pyplot as plt
import climage
from torchwindow import Window
import os
import time

FLOAT_EPSILON = wp.constant(sys.float_info.epsilon)

from pxr import Usd, UsdGeom, UsdSkel, Sdf, Vt, Gf

class RenderMode:
    """Rendering modes
    grayscale: lambertian shading from multiple directional lights
    texture: 2D texture map
    normal_map: mesh normal computed from interpolated vertex normals
    """
    grayscale = 0
    texture = 1
    normal_map = 2
    vertex_color = 3
    depth_map = 4

@wp.struct
class RenderMesh:
    """Mesh to be ray traced.
    Assumes a triangle mesh as input.
    Per-vertex normals are computed with compute_vertex_normals()
    """
    # id: wp.uint64
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=int)
    texCoords: wp.array(dtype=wp.vec2)
    texCoordsShift: wp.array(dtype=wp.int32)
    texIndices: wp.array(dtype=wp.int32)
    texIndicesShift: wp.array(dtype=wp.int32)
    texture: wp.array(dtype=wp.vec3)
    textureShift: wp.array(dtype=wp.int32)
    textureSize: wp.array(dtype=wp.int32)
    vertexNormals: wp.array(dtype=wp.vec3)
    vertexColors: wp.array(dtype=wp.vec3)
    pos: wp.array(dtype=wp.vec3)
    rot: wp.array(dtype=wp.quat)
    numTris: wp.int32
    visFaceToObjectId: wp.array(dtype=wp.int32)
    objectUsesTexture: wp.array(dtype=wp.int32)
    objectNumVisFaceCummulative: wp.array(dtype=wp.int32)

@wp.struct
class Camera:
    """Basic camera for ray tracing
    """
    horizontal: float
    vertical: float
    aspect: float
    e: float
    tan: float
    pos: wp.array(dtype=wp.vec3)
    at: wp.vec3
    rot: wp.array(dtype=wp.quat)

@wp.struct        
class DirectionalLights:
    """Stores arrays of directional light directions and intensities.
    """
    dirs: wp.array(dtype=wp.vec3)
    intensities: wp.array(dtype=float)
    num_lights: int

@wp.struct
class MeshQueryResult:
    hitVec: wp.vec3
    triId: wp.int32
    t: wp.float32

@wp.kernel
def vertex_normal_sum_kernel(
        verts: wp.array(dtype=wp.vec3),
        indices: wp.array(dtype=int),
        normal_sums: wp.array(dtype=wp.vec3)
        ):

    tid = wp.tid()

    i = indices[tid*3]
    j = indices[tid*3 + 1]
    k = indices[tid*3 + 2]

    a = verts[i]
    b = verts[j]
    c = verts[k]

    ab = b - a
    ac = c - a

    area_normal = wp.cross(ab, ac)

    wp.atomic_add(normal_sums, i, area_normal)
    wp.atomic_add(normal_sums, j, area_normal)
    wp.atomic_add(normal_sums, k, area_normal)

@wp.kernel
def zeroVec3Array(array: wp.array(dtype=wp.vec3)):

    tid = wp.tid()
    array[tid] = wp.vec3(0.0, 0.0, 0.0)

@wp.kernel
def normalize_kernel(
        normal_sums: wp.array(dtype=wp.vec3),
        vertex_normals: wp.array(dtype=wp.vec3),
        ):

    tid = wp.tid()

    vertex_normals[tid] = wp.normalize(normal_sums[tid])

@wp.func
def texture_interpolation(
        tex_interp: wp.vec2,
        texture: wp.array(dtype=wp.vec3),
        textureShift: wp.int32,
        textureSize: wp.int32):
    
    tex_width = textureSize
    tex_height = textureSize
    tex = wp.vec2(tex_interp[0] * float(tex_width - 1), (1.0 - tex_interp[1]) * float(tex_height - 1))
    
    x0 = int(tex[0])
    x1 = x0 + 1
    alpha_x = tex[0] - float(x0)
    y0 = int(tex[1])
    y1 = y0 + 1
    alpha_y = tex[1] - float(y0)
    c00 = texture[textureShift + y0 + x0 * tex_width]
    c10 = texture[textureShift + y0 + x1 * tex_width]
    c01 = texture[textureShift + y1 + x0 * tex_width]
    c11 = texture[textureShift + y1 + x1 * tex_width]
    lower = (1.0 - alpha_x) * c00 + alpha_x * c10
    upper = (1.0 - alpha_x) * c01 + alpha_x * c11
    color = (1.0 - alpha_y) * lower + alpha_y * upper

    return color

@wp.kernel
def drawSimBVHKernel(mesh: RenderMesh,
                     camera: Camera,
                     rays_width: int,
                     rays_height: int,
                     rays: wp.array(dtype=wp.vec3),
                     lights: DirectionalLights,
                     mode: int,
                     aabbMin: wp.array(dtype=wp.vec3),
                     aabbMax: wp.array(dtype=wp.vec3),
                     leftNode: wp.array(dtype=wp.int32),
                     triMap: wp.array(dtype=wp.int32),
                     firstTriId: wp.array(dtype=wp.int32),
                     triCount: wp.array(dtype=wp.int32),
                     numBVHNodes: wp.int32,
                     stack: wp.array(dtype=wp.int32),
                     currentEnv: wp.int32,
                     numEnv: wp.int32,
                     numEnvMeshVisFaces: wp.int32,
                     numEnvRigidVisFaces: wp.int32,
                     numEnvLaparoscopeVisFaces: wp.int32):
    
    tid = wp.tid()
    
    x = tid % rays_width
    y = rays_height - tid // rays_width
    
    sx = 2.0*float(x)/float(rays_width) - 1.0
    sy = 2.0*float(y)/float(rays_height) - 1.0

    # compute view ray in world space
    ro_world = camera.pos[0]
    rd_world = wp.normalize(wp.quat_rotate(camera.rot[0], wp.vec3(sx * camera.tan * camera.aspect, sy * camera.tan, -1.0)))

    # compute view ray in mesh space
    inv = wp.transform_inverse(wp.transform(mesh.pos[0], mesh.rot[0]))
    ro = wp.transform_point(inv, ro_world)
    rd = wp.transform_vector(inv, rd_world)

    # color = wp.vec3(1.0, 1.0, 1.0)
    color = wp.vec3(0.0, 0.0, 0.0)
    
    meshQueryRes = meshQueryRay(ro, rd, mesh, aabbMin, aabbMax, leftNode, triMap, firstTriId, triCount, numBVHNodes, stack, tid, currentEnv, numEnv, numEnvMeshVisFaces, numEnvRigidVisFaces, numEnvLaparoscopeVisFaces)
    # triP = meshQueryRay(ro, rd, mesh, aabbMin, aabbMax, leftNode, triMap, firstTriId, triCount, stack, tid, envNodeShift, envTriShift)
    
    triId = wp.int32(meshQueryRes[0][0])
    localTriId = wp.int32(meshQueryRes[2][0])
    p = wp.vec3(meshQueryRes[1][0], meshQueryRes[1][1], meshQueryRes[1][2])
    t = meshQueryRes[0][1]

    if triId > -1:

        i = mesh.indices[triId*3]
        j = mesh.indices[triId*3 + 1]
        k = mesh.indices[triId*3 + 2]

        a = mesh.vertices[i]
        b = mesh.vertices[j]
        c = mesh.vertices[k]

        # barycentric coordinates
        tri_area = wp.length(wp.cross(b-a, c-a))
        w = wp.length(wp.cross(b-a, p-a)) / tri_area
        v = wp.length(wp.cross(p-a, c-a)) / tri_area
        u = 1.0 - w - v

        a_n = mesh.vertexNormals[i]
        b_n = mesh.vertexNormals[j]
        c_n = mesh.vertexNormals[k]

        # vertex normal interpolation
        normal = u * a_n + v * b_n + w * c_n

        if mode == 0 or mode == 1 or mode == 3:
        
            if mode == 0:  # grayscale
                color = wp.vec3(1.0)

            elif mode == 1:  # texture interpolation

                objectId = mesh.visFaceToObjectId[triId]
                textureId = mesh.objectUsesTexture[objectId]

                if textureId > 0:
                    textureId -= 1
                    texturesShift = mesh.textureShift[textureId]
                    texturesSize = mesh.textureSize[textureId]
                    texCoordsShift = mesh.texCoordsShift[textureId]
                    texIndicesShift = mesh.texIndicesShift[textureId]
                    tex_a = mesh.texCoords[texCoordsShift + mesh.texIndices[texIndicesShift + localTriId * 3]]
                    tex_b = mesh.texCoords[texCoordsShift + mesh.texIndices[texIndicesShift + localTriId * 3 + 1]]
                    tex_c = mesh.texCoords[texCoordsShift + mesh.texIndices[texIndicesShift + localTriId * 3 + 2]]

                    tex = u * tex_a + v * tex_b + w * tex_c

                    color = texture_interpolation(tex, mesh.texture, texturesShift, texturesSize)

            elif mode == 3:

                color = (mesh.vertexColors[i] + mesh.vertexColors[j] + mesh.vertexColors[k]) / 3.0

            # lambertian directional lighting
            lambert = float(0.0)
            for i in range(lights.num_lights):
                dir = wp.transform_vector(inv, lights.dirs[i])
                val = lights.intensities[i] * wp.dot(normal, dir)
                if val < 0.0:
                    val = 0.0
                lambert = lambert + val

            color = lambert * color

        elif mode == 2:  # normal map
        
            color = normal * 0.5 + wp.vec3(0.5, 0.5, 0.5)

        elif mode == 4:  # depth map
        
            color = t / 50.0 * wp.vec3(1.0, 1.0, 1.0)

        if (color[0] > 1.0): color = wp.vec3(1.0, color[1], color[2])
        if (color[1] > 1.0): color = wp.vec3(color[0], 1.0, color[2])
        if (color[2] > 1.0): color = wp.vec3(color[0], color[1], 1.0)

    rays[tid] = color

@wp.kernel
def drawSimBVHKernelManyEnvs(mesh: RenderMesh,
                             camera: Camera,
                             rays_width: int,
                             rays_height: int,
                             rays: wp.array(dtype=wp.vec3),
                             lights: DirectionalLights,
                             mode: int,
                             aabbMin: wp.array(dtype=wp.vec3),
                             aabbMax: wp.array(dtype=wp.vec3),
                             leftNode: wp.array(dtype=wp.int32),
                             triMap: wp.array(dtype=wp.int32),
                             firstTriId: wp.array(dtype=wp.int32),
                             triCount: wp.array(dtype=wp.int32),
                             numBVHNodes: wp.int32,
                             stack: wp.array(dtype=wp.int32),
                             numEnv: wp.int32,
                             numEnvMeshVisFaces: wp.int32,
                             numEnvRigidVisFaces: wp.int32,
                             numEnvLaparoscopeVisFaces: wp.int32):
    
    tid = wp.tid()

    currentEnv = tid // (rays_width * rays_height)
    envTid = tid % (rays_width * rays_height)
    
    x = envTid % rays_width
    y = rays_height - envTid // rays_width
    
    sx = 2.0*float(x)/float(rays_width) - 1.0
    sy = 2.0*float(y)/float(rays_height) - 1.0

    # compute view ray in world space
    ro_world = camera.pos[0]
    rd_world = wp.normalize(wp.quat_rotate(camera.rot[0], wp.vec3(sx * camera.tan * camera.aspect, sy * camera.tan, -1.0)))

    # compute view ray in mesh space
    inv = wp.transform_inverse(wp.transform(mesh.pos[0], mesh.rot[0]))
    ro = wp.transform_point(inv, ro_world)
    rd = wp.transform_vector(inv, rd_world)

    # color = wp.vec3(1.0, 1.0, 1.0)
    color = wp.vec3(0.0, 0.0, 0.0)
    
    meshQueryRes = meshQueryRayManyEnvs(ro, rd, mesh, aabbMin, aabbMax, leftNode, triMap, firstTriId, triCount, numBVHNodes, stack, tid, currentEnv, numEnv, numEnvMeshVisFaces, numEnvRigidVisFaces, numEnvLaparoscopeVisFaces)
    
    triId = wp.int32(meshQueryRes[0][0])
    localEnvTriId = wp.int32(meshQueryRes[2][0])

    p = wp.vec3(meshQueryRes[1][0], meshQueryRes[1][1], meshQueryRes[1][2])
    t = meshQueryRes[0][1]

    if triId > -1:

        i = mesh.indices[triId * 3]
        j = mesh.indices[triId * 3 + 1]
        k = mesh.indices[triId * 3 + 2]

        a = mesh.vertices[i]
        b = mesh.vertices[j]
        c = mesh.vertices[k]

        # barycentric coordinates
        tri_area = wp.length(wp.cross(b-a, c-a))
        w = wp.length(wp.cross(b-a, p-a)) / tri_area
        v = wp.length(wp.cross(p-a, c-a)) / tri_area
        u = 1.0 - w - v

        a_n = mesh.vertexNormals[i]
        b_n = mesh.vertexNormals[j]
        c_n = mesh.vertexNormals[k]

        # vertex normal interpolation
        normal = u * a_n + v * b_n + w * c_n

        if mode == 0 or mode == 1 or mode == 3:
        
            if mode == 0:  # grayscale
                color = wp.vec3(1.0)

            elif mode == 1:  # texture interpolation
                
                objectId = mesh.visFaceToObjectId[localEnvTriId]
                textureId = mesh.objectUsesTexture[objectId]

                if textureId > 0:
                    localTriId = localEnvTriId - mesh.objectNumVisFaceCummulative[objectId]
                    textureId -= 1

                    texturesShift = mesh.textureShift[textureId]
                    texturesSize = mesh.textureSize[textureId]
                    texCoordsShift = mesh.texCoordsShift[textureId]
                    texIndicesShift = mesh.texIndicesShift[textureId]

                    tex_a = mesh.texCoords[texCoordsShift + mesh.texIndices[texIndicesShift + localTriId * 3]]
                    tex_b = mesh.texCoords[texCoordsShift + mesh.texIndices[texIndicesShift + localTriId * 3 + 1]]
                    tex_c = mesh.texCoords[texCoordsShift + mesh.texIndices[texIndicesShift + localTriId * 3 + 2]]

                    tex = u * tex_a + v * tex_b + w * tex_c

                    color = texture_interpolation(tex, mesh.texture, texturesShift, texturesSize)

                # # revert to vertex color if no texture for this mesh
                else:
                    color = (mesh.vertexColors[i] + mesh.vertexColors[j] + mesh.vertexColors[k]) / 3.0

            elif mode == 3:

                color = (mesh.vertexColors[i] + mesh.vertexColors[j] + mesh.vertexColors[k]) / 3.0

            # lambertian directional lighting
            lambert = float(0.0)
            for i in range(lights.num_lights):
                dir = wp.transform_vector(inv, lights.dirs[i])
                val = lights.intensities[i] * wp.dot(normal, dir)
                if val < 0.0:
                    val = 0.0
                lambert = lambert + val

            color = lambert * color

        elif mode == 2:  # normal map
        
            color = normal * 0.5 + wp.vec3(0.5, 0.5, 0.5)

        elif mode == 4:  # depth map
        
            color = t / 50.0 * wp.vec3(1.0, 1.0, 1.0)

        if (color[0] > 1.0): color = wp.vec3(1.0, color[1], color[2])
        if (color[1] > 1.0): color = wp.vec3(color[0], 1.0, color[2])
        if (color[2] > 1.0): color = wp.vec3(color[0], color[1], 1.0)

    rays[tid] = color

@wp.func
def meshQueryRayManyEnvs(ro: wp.vec3,
                         rd: wp.vec3,
                         mesh: RenderMesh,
                         aabbMin: wp.array(dtype=wp.vec3),
                         aabbMax: wp.array(dtype=wp.vec3),
                         leftNode: wp.array(dtype=wp.int32),
                         triMap: wp.array(dtype=wp.int32),
                         firstTriId: wp.array(dtype=wp.int32),
                         triCount: wp.array(dtype=wp.int32),
                         numBVHNodes: wp.int32,
                         stack: wp.array(dtype=wp.int32),
                         tid: wp.int32,
                         currentEnv: wp.int32,
                         numEnv: wp.int32,
                         numEnvMeshVisFaces: wp.int32,
                         numEnvRigidVisFaces: wp.int32,
                         numEnvLaparoscopeVisFaces: wp.int32) -> wp.mat33f:

    stackShift = tid * 32

    stack[stackShift] = 0
    count = wp.int32(1)

    meshQueryRes = wp.mat33f(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    tMin = wp.float32(1e6)
    tMax = 1e6

    aabbShift = currentEnv * numBVHNodes

    while count:
        nodeId = stack[stackShift + (count - 1)]
        count -= 1

        lower = aabbMin[nodeId + aabbShift]
        upper = aabbMax[nodeId + aabbShift]

        hit = intersectAABB(ro, rd, lower, upper)

        if hit[0] == 1.0 and hit[1] < tMin:
            leftId = leftNode[nodeId]
            rightId = leftId + 1

            if leftId == -1:
                firstTri = firstTriId[nodeId]
                for nextTri in range(triCount[nodeId]):
                
                    triId = triMap[firstTri + nextTri]
                    
                    # ToDo: This should be moved to a mapping array
                    visFaceShift = 0
                    localEnvTriId = triId

                    if triId >= (numEnvMeshVisFaces + numEnvRigidVisFaces):
                        visFaceShift = (numEnv-1) * (numEnvMeshVisFaces + numEnvRigidVisFaces) + currentEnv * numEnvLaparoscopeVisFaces
                    elif triId >= numEnvMeshVisFaces:
                        visFaceShift = (numEnv-1) * numEnvMeshVisFaces + currentEnv * numEnvRigidVisFaces
                    else:
                        visFaceShift = currentEnv * numEnvMeshVisFaces

                    triId = triId + visFaceShift

                    i = mesh.indices[triId*3]
                    j = mesh.indices[triId*3 + 1]
                    k = mesh.indices[triId*3 + 2]

                    a = mesh.vertices[i]
                    b = mesh.vertices[j]
                    c = mesh.vertices[k]

                    # boolVec = checkTriRayIntersection(a, b, c, ro, rd, tid)
                    boolVec = MTTriRayIntersection(a, b, c, ro, rd)
                    tTriangle = boolVec[0]

                    if tTriangle < tMin and tTriangle > 0.0:

                        tMin = tTriangle
                        meshQueryRes = wp.mat33f(float(triId), tMin, 0.0, boolVec[1], boolVec[2], boolVec[3], float(localEnvTriId), 0.0, 0.0)
            else:
                stack[stackShift + (count)] = leftId
                stack[stackShift + (count + 1)] = rightId
                count += 2

    if tMin >= tMax:
        meshQueryRes = wp.mat33f(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    return meshQueryRes

@wp.func
def meshQueryRay(ro: wp.vec3,
                 rd: wp.vec3,
                 mesh: RenderMesh,
                 aabbMin: wp.array(dtype=wp.vec3),
                 aabbMax: wp.array(dtype=wp.vec3),
                 leftNode: wp.array(dtype=wp.int32),
                 triMap: wp.array(dtype=wp.int32),
                 firstTriId: wp.array(dtype=wp.int32),
                 triCount: wp.array(dtype=wp.int32),
                 numBVHNodes: wp.int32,
                 stack: wp.array(dtype=wp.int32),
                 tid: wp.int32,
                 currentEnv: wp.int32,
                 numEnv: wp.int32,
                 numEnvMeshVisFaces: wp.int32,
                 numEnvRigidVisFaces: wp.int32,
                 numEnvLaparoscopeVisFaces: wp.int32) -> wp.mat33f:

    stackShift = tid * 32

    stack[stackShift + 0] = 0
    count = wp.int32(1)

    meshQueryRes = wp.mat33f(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    tMin = wp.float32(1e6)
    tMax = 1e6

    aabbShift = currentEnv * numBVHNodes

    while count:
        nodeId = stack[stackShift + (count - 1)]
        count -= 1

        lower = aabbMin[nodeId + aabbShift]
        upper = aabbMax[nodeId + aabbShift]

        hit = intersectAABB(ro, rd, lower, upper)

        if hit[0] == 1.0 and hit[1] < tMin:
            leftId = leftNode[nodeId]
            rightId = leftId + 1

            if leftId == -1:
                firstTri = firstTriId[nodeId]
                for nextTri in range(triCount[nodeId]):
                
                    triId = triMap[firstTri + nextTri]
                    
                    # ToDo: This should be moved to a mapping array
                    visFaceShift = 0

                    if triId >= (numEnvMeshVisFaces + numEnvRigidVisFaces):
                        visFaceShift = (numEnv-1) * (numEnvMeshVisFaces + numEnvRigidVisFaces) + currentEnv * numEnvLaparoscopeVisFaces
                    elif triId >= numEnvMeshVisFaces:
                        visFaceShift = (numEnv-1) * numEnvMeshVisFaces + currentEnv * numEnvRigidVisFaces
                    else:
                        visFaceShift = currentEnv * numEnvMeshVisFaces

                    triId = triId + visFaceShift

                    i = mesh.indices[triId*3]
                    j = mesh.indices[triId*3 + 1]
                    k = mesh.indices[triId*3 + 2]

                    a = mesh.vertices[i]
                    b = mesh.vertices[j]
                    c = mesh.vertices[k]

                    # boolVec = checkTriRayIntersection(a, b, c, ro, rd, tid)
                    boolVec = MTTriRayIntersection(a, b, c, ro, rd)
                    tTriangle = boolVec[0]

                    if tTriangle < tMin and tTriangle > 0.0:

                        tMin = tTriangle
                        meshQueryRes = wp.mat33f(float(triId), tMin, 0.0, boolVec[1], boolVec[2], boolVec[3], 0.0, 0.0, 0.0)
            else:
                stack[stackShift + (count)] = leftId
                stack[stackShift + (count + 1)] = rightId
                count += 2

    if tMin >= tMax:
        meshQueryRes = wp.mat33f(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    return meshQueryRes

@wp.func
def intersectAABB(rO: wp.vec3,
                  rD: wp.vec3,
                  aabbMin: wp.vec3,
                  aabbMax: wp.vec3) -> wp.vec2:
    
    t1 = (aabbMin[0] - rO[0]) / rD[0]
    t2 = (aabbMax[0] - rO[0]) / rD[0]
    t3 = (aabbMin[1] - rO[1]) / rD[1]
    t4 = (aabbMax[1] - rO[1]) / rD[1]
    t5 = (aabbMin[2] - rO[2]) / rD[2]
    t6 = (aabbMax[2] - rO[2]) / rD[2]
    tmin = wp.max(wp.max(wp.min(t1, t2), wp.min(t3, t4)), wp.min(t5, t6))
    tmax = wp.min(wp.min(wp.max(t1, t2), wp.max(t3, t4)), wp.max(t5, t6))

    hit = 0.0
    t = 0.0
    if (tmax >= 0.0) and (tmax >= tmin):
        hit = 1.0
        t = tmin

    return wp.vec2(hit, t)

@wp.func
def checkTriRayIntersection(a: wp.vec3,
                            b: wp.vec3,
                            c: wp.vec3,
                            ro: wp.vec3,
                            rd: wp.vec3,
                            tid: wp.int32) -> wp.vec4:
    pAB = b - a
    pAC = c - a

    # Triangle normal
    N = wp.cross(pAB, pAC)
    # triArea = wp.length(N)

    retValue = wp.vec4()

    # Check if ray and plane parallel
    NDotRayDirection = wp.dot(N, rd)
    if (wp.abs(NDotRayDirection) < 0.0001):
        return retValue

    d = -wp.dot(N, a)
    t = -(wp.dot(N, ro) + d) / NDotRayDirection

    if(t < 0.0):
        return retValue

    p = ro + t * rd

    edge0 = b - a
    ap = p - a
    C = wp.cross(edge0, ap)
    if(wp.dot(N, C) < 0):
        return retValue

    edge1 = c - b
    bp = p - b
    C = wp.cross(edge1, bp)
    if(wp.dot(N, C) < 0):
        return retValue

    edge2 = a - c
    cp = p - c
    C = wp.cross(edge2, cp)
    if(wp.dot(N, C) < 0):
        return retValue
    
    retValue = wp.vec4(1.0, p[0], p[1], p[2])
    return retValue

# from https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
@wp.func
def MTTriRayIntersection(a: wp.vec3,
                         b: wp.vec3,
                         c: wp.vec3,
                         ro: wp.vec3,
                         rd: wp.vec3) -> wp.vec4:
    pAB = b - a
    pAC = c - a

    pvec = wp.cross(rd, pAC)
    det = wp.dot(pAB, pvec)

    retValue = wp.vec4(-1.0, 0.0, 0.0, 0.0)

    # Check if ray and plane parallel
    if (wp.abs(det) < 0.0001):
        return retValue

    invDet = 1.0 / det
    tvec = ro - a
    u = wp.dot(tvec, pvec) * invDet

    if u < 0.0 or u > 1.0:
        return retValue
    
    qvec = wp.cross(tvec, pAB)
    v = wp.dot(rd, qvec) * invDet
    if v < 0.0 or u + v > 1.0:
        return retValue

    t = wp.dot(pAC, qvec) * invDet
    # We can return only t
    p = ro + t * rd

    retValue = wp.vec4(t, p[0], p[1], p[2])
    return retValue

@wp.kernel
def downsample_kernel(
        rays: wp.array(dtype=wp.vec3),
        pixels: wp.array(dtype=wp.vec3),
        rays_width: int,
        num_samples: int):

    tid = wp.tid()

    pixels_width = rays_width / num_samples
    px = tid % pixels_width
    py = tid // pixels_width
    start_idx = py * num_samples * rays_width + px * num_samples

    # color = wp.vec3(1.0, 1.0, 1.0)
    color = wp.vec3(0.0, 0.0, 0.0)

    for i in range(0, num_samples):
        for j in range(0, num_samples):
            ray = rays[start_idx + i * rays_width + j]
            color = wp.vec3(color[0] + ray[0], color[1] + ray[1], color[2] + ray[2])

    num_samples_sq = float(num_samples * num_samples)
    color = wp.vec3(color[0] / num_samples_sq, color[1] / num_samples_sq, color[2] / num_samples_sq)

    pixels[tid] = color

def _usd_add_xform(prim):

    from pxr import UsdGeom

    prim = UsdGeom.Xform(prim)
    prim.ClearXformOpOrder()

    t = prim.AddTranslateOp()
    r = prim.AddOrientOp()
    s = prim.AddScaleOp()

def _usd_set_xform(xform, pos: tuple, rot: tuple, scale: tuple, time):

    from pxr import UsdGeom, Gf

    xform = UsdGeom.Xform(xform)
    
    xform_ops = xform.GetOrderedXformOps()

    xform_ops[0].Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])), time)
    xform_ops[1].Set(Gf.Quatf(float(rot[3]), float(rot[0]), float(rot[1]), float(rot[2])), time)
    xform_ops[2].Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])), time)

def _usd_set_transform(xform, transformMat, time):

    xform = UsdGeom.Xform(xform)
    xform.ClearXformOpOrder()
    
    transform_op = None
    for xformOp in xform.GetOrderedXformOps():
        if xformOp.GetOpType() == UsdGeom.XformOp.TypeTransform:
            transform_op = xformOp
    if transform_op:
        xform_op = transform_op
    else:
        xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
    
    xform_op.Set(transformMat, time)

# transforms a cylinder such that it connects the two points pos0, pos1
def _compute_segment_xform(pos0, pos1):

    from pxr import Gf

    mid = (pos0 + pos1) * 0.5
    height = (pos1 - pos0).GetLength()

    dir = (pos1 - pos0) / height

    rot = Gf.Rotation()
    rot.SetRotateInto((0.0, 0.0, 1.0), Gf.Vec3d(dir))

    scale = Gf.Vec3f(1.0, 1.0, height)

    return (mid, Gf.Quath(rot.GetQuat()), scale)

def bourke_color_map(low, high, v):

	c = [1.0, 1.0, 1.0]

	if v < low:
		v = low
	if v > high:
		v = high
	dv = high - low

	if v < (low + 0.25 * dv):
		c[0] = 0.
		c[1] = 4. * (v - low) / dv
	elif v < (low + 0.5 * dv):
		c[0] = 0.
		c[2] = 1. + 4. * (low + 0.25 * dv - v) / dv
	elif v < (low + 0.75 * dv):
		c[0] = 4. * (v - low - 0.5 * dv) / dv
		c[2] = 0.
	else:
		c[1] = 1. + 4. * (low + 0.75 * dv - v) / dv
		c[2] = 0.

	return c

def getGfMatrix(matrix):
    
    transformation = Gf.Matrix4d(matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3],
                                 matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3],
                                 matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3],
                                 matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3])
    
    return transformation

def setXform(xform, transformation, time, scale=Gf.Vec3d(1.0, 1.0, 1.0)):
    xform_ops = xform.GetOrderedXformOps()
    
    pos = transformation.ExtractTranslation()

    rot = Gf.Quatf(transformation.ExtractRotation().GetQuat())
    xform_ops[0].Set(pos, time)
    xform_ops[1].Set(rot, time)
    xform_ops[2].Set(scale, time)

class WarpRaycastRendererDO:

    def __init__(self, device, simModel, cameraPos=[3.0, 3.0, 3.0], cameraRot=[0.0, 0.0, 0.0], lightPos=[3.0, 0.0, 0.0], lightIntensity=0.3, resolution:int=512, mode="human", horizontalAperture=19.2, verticalAperture=19.2):

        self.device=device
        self.mode = mode
        self.simModel = simModel
        self.numEnvs = simModel.numEnvs
        self.renderGraph = None

        self.cameraPos = wp.vec3(cameraPos[0], cameraPos[1], cameraPos[2])
        self.cameraRot = wp.vec3(cameraRot[0], cameraRot[1], cameraRot[2])

        horizontal_aperture = horizontalAperture
        vertical_aperture = verticalAperture
        aspect = horizontal_aperture / vertical_aperture
        focal_length = 50.0
        self.height = resolution
        self.width = resolution
        # self.width = int(aspect * self.height)
        self.num_pixels = self.width * self.height

        # set anti-aliasing
        self.num_samples = 1

        # set render mode
        self.render_mode = RenderMode.texture
        # self.render_mode = RenderMode.grayscale
        # self.render_mode = RenderMode.vertex_color
        # self.render_mode = RenderMode.normal_map
        # self.render_mode = RenderMode.depth_map

        # construct camera
        self.camera = Camera()
        self.camera.horizontal = horizontal_aperture
        self.camera.vertical = vertical_aperture
        self.camera.aspect = aspect
        self.camera.e = focal_length
        self.camera.tan = vertical_aperture / (2.0 * focal_length)
        self.camera.pos = wp.array(self.cameraPos, dtype=wp.vec3, device=self.device)
        self.camera.rot = wp.array(wp.quat_rpy(self.cameraRot[2], self.cameraRot[0], self.cameraRot[1]), dtype=wp.quat, device=self.device)
        # self.camera.at = wp.normalize(wp.quat_rotate(self.camera.rot, wp.vec3(0.0, 0.0, -1.0)))

        # construct lights
        self.lights = DirectionalLights()
        self.lights.dirs = wp.array(np.array([lightPos]), dtype=wp.vec3, device=self.device)
        self.lights.intensities = wp.array(np.array([lightIntensity]), dtype=float, device=self.device)
        self.lights.num_lights = 1

        # construct rays
        self.rays_width = self.width * pow(2, self.num_samples)
        self.rays_height = self.height * pow(2, self.num_samples)
        self.num_rays = self.rays_width * self.rays_height
        # Switch on for sequential rendering
        # self.rays = wp.zeros(self.num_rays, dtype=wp.vec3, device=self.device)
        self.rays = wp.zeros(self.num_rays * self.numEnvs, dtype=wp.vec3, device=self.device)

        # construct pixels
        # Switch on for sequential rendering
        # self.pixels = wp.zeros(self.num_pixels, dtype=wp.vec3, device=self.device)
        self.pixels = wp.zeros(self.num_pixels * self.numEnvs, dtype=wp.vec3, device=self.device)

        # create plot window
        if self.mode in ["human", "debug"]:
            plt.figure(figsize=(10, 10))
            plt.axis("off")
        if self.mode == "gpu":
        # if self.mode in ["gpu", "gpudebug"]:
            numEnvsVertical = math.floor(math.sqrt(self.simModel.numEnvs))
            numEnvsHorizontal = math.ceil(self.simModel.numEnvs/numEnvsVertical)
            self.window = Window(resolution*numEnvsHorizontal, resolution*numEnvsVertical, "diffKit")
            # self.window = Window(resolution, resolution, "diffKit")
        self.img = None

        # construct meshes
        self.renderMesh = RenderMesh()

        self.renderMesh.vertices = simModel.allVisPoint
        self.renderMesh.indices = simModel.allVisFace
        self.renderMesh.texCoords = simModel.texCoords
        self.renderMesh.texCoordsShift = simModel.texCoordsShift
        self.renderMesh.texIndices = simModel.texIndices
        self.renderMesh.texIndicesShift = simModel.texIndicesShift
        self.renderMesh.texture = simModel.textures
        self.renderMesh.textureShift = simModel.texturesShift
        self.renderMesh.textureSize = simModel.texturesSize
        self.renderMesh.objectUsesTexture = simModel.objectUsesTexture
        self.renderMesh.visFaceToObjectId = simModel.visFaceToObjectId
        self.renderMesh.objectNumVisFaceCummulative = simModel.objectNumVisFaceCummulative
        self.renderMesh.vertexColors = simModel.allVisPointColor
        self.meshNormalSums = wp.zeros(simModel.numAllVisPoints, dtype=wp.vec3, device=self.device)
        self.renderMesh.vertexNormals = wp.zeros(simModel.numAllVisPoints, dtype=wp.vec3, device=self.device)
        # self.renderMesh.pos = simMesh.translationWP
        self.renderMesh.pos = wp.zeros(1, dtype=wp.vec3, device=self.device)
        # self.renderMesh.rot = simMesh.rotationWP
        self.renderMesh.rot = wp.array(np.array([0.0, 0.0, 0.0, 1.0]), dtype=wp.quat, device=self.device)
        self.renderMesh.numTris = len(self.renderMesh.indices)

        self.meshStack = wp.zeros(self.num_rays * 32 * self.numEnvs, dtype=wp.int32, device=self.device)

    def renderFunction(self, simBVH, currentEnv=0):

        wp.launch(kernel=zeroVec3Array,
            dim=self.simModel.numAllVisPoints,
            inputs=[self.meshNormalSums])

        # compute vertex normals
        wp.launch(kernel=vertex_normal_sum_kernel,
            dim=self.simModel.numAllVisFaces,
            inputs=[self.renderMesh.vertices, self.renderMesh.indices, self.meshNormalSums])
        
        wp.launch(kernel=normalize_kernel,
            dim=self.simModel.numAllVisPoints,
            inputs=[self.meshNormalSums, self.renderMesh.vertexNormals])

        wp.launch(kernel=drawSimBVHKernel,
            dim=self.num_rays,
            inputs=[
                self.renderMesh,
                self.camera,
                self.rays_width,
                self.rays_height,
                self.rays,
                self.lights,
                self.render_mode,
                simBVH.aabbMin,
                simBVH.aabbMax,
                simBVH.leftNode,
                simBVH.triId,
                simBVH.firstTriId,
                simBVH.triCount,
                simBVH.numNodes,
                self.meshStack,
                currentEnv,
                self.simModel.numEnvs,
                self.simModel.numEnvMeshesVisFaces,
                self.simModel.numEnvRigidsVisFaces,
                self.simModel.numEnvLaparoscopeVisFaces
            ])
        
        # downsample
        wp.launch(kernel=downsample_kernel,
            dim=self.num_pixels,
            inputs=[self.rays,
                    self.pixels,
                    self.rays_width,
                    pow(2, self.num_samples)]
        )

    def renderFunctionNew(self, simBVH):

        # Needed for correct normals and therefore colors rendering
        wp.launch(kernel=zeroVec3Array,
            dim=self.simModel.numAllVisPoints,
            inputs=[self.meshNormalSums])

        # compute vertex normals
        wp.launch(kernel=vertex_normal_sum_kernel,
            dim=self.simModel.numAllVisFaces,
            inputs=[self.renderMesh.vertices, self.renderMesh.indices, self.meshNormalSums])
        
        wp.launch(kernel=normalize_kernel,
            dim=self.simModel.numAllVisPoints,
            inputs=[self.meshNormalSums, self.renderMesh.vertexNormals])

        wp.launch(kernel=drawSimBVHKernelManyEnvs,
            dim=self.num_rays * self.numEnvs,
            inputs=[
                self.renderMesh,
                self.camera,
                self.rays_width,
                self.rays_height,
                self.rays,
                self.lights,
                self.render_mode,
                simBVH.aabbMin,
                simBVH.aabbMax,
                simBVH.leftNode,
                simBVH.triId,
                simBVH.firstTriId,
                simBVH.triCount,
                simBVH.numNodes,
                self.meshStack,
                self.simModel.numEnvs,
                self.simModel.numEnvMeshesVisFaces,
                self.simModel.numEnvRigidsVisFaces,
                self.simModel.numEnvLaparoscopeVisFaces
            ])
        
        # downsample
        wp.launch(kernel=downsample_kernel,
            dim=self.num_pixels * self.numEnvs,
            inputs=[self.rays,
                    self.pixels,
                    self.rays_width,
                    pow(2, self.num_samples)]
        )

    def render(self, simBVH, simModel, useGraph=False):
        images = None
        imagesHeadless = None
        imageRows = []
        sqrtNumEnvs = int(math.sqrt(simModel.numEnvs))

        with wp.ScopedDevice("cuda:0"):

            with wp.ScopedTimer("Render", active=False, detailed=False):

                for i in range(simModel.numEnvs):
                    self.pixels = wp.zeros(self.num_pixels, dtype=wp.vec3, device=self.device)

                    if useGraph:
                        raise Exception("Sorry, didn't find a way to implement this yet")
                        # if self.renderGraph == None:
                        #     wp.capture_begin()
                        #     self.renderFunction(simBVH, i)
                        #     self.renderGraph = wp.capture_end()
                        
                        # wp.capture_launch(self.renderGraph)

                    else:
                        self.renderFunction(simBVH, i)

                    # image = wp.to_torch(self.pixels).view(self.height, self.width, 3)
                    image = torch.clone(wp.to_torch(self.pixels).view(self.height, self.width, 3))

                    if self.mode in ["human", "debug", "terminal", "gpu", "gpuless"]:
                        rowId = int(i / sqrtNumEnvs)
                        columnId = i % sqrtNumEnvs

                        # arrange in 2D
                        if columnId == 0:
                            imageRows.append(image)
                        else:
                            imageRows[rowId] = torch.cat((imageRows[rowId], image), 0)
                    
                    if self.mode in ["headless", "debug", "gpu", "gpuless"]:
                        if imagesHeadless is None:
                            # only RGB channel
                            imagesHeadless = image[None]
                        else:
                            imagesHeadless = torch.cat((imagesHeadless, image[None]), 0)
                
            if self.mode in ["human", "debug", "terminal", "gpu", "gpuless"]:
                lastRowShape = imageRows[-1].shape
                if not imageRows[0].shape == lastRowShape:
                    missingImages = torch.ones((imageRows[0].shape[0] - lastRowShape[0], lastRowShape[1], lastRowShape[2]), device=self.device)
                    imageRows[-1] = torch.cat((imageRows[-1], missingImages), 0)
                images = torch.cat((imageRows), 1)
        
        if self.mode in ["human", "debug", "terminal"]:
            if self.img is None:
                plt.axis('off')
                self.img = plt.imshow(images.cpu().numpy())
            else:
                self.img.set_data(images.cpu().numpy())

            if self.mode == "terminal":
                plt.savefig("fig.png", bbox_inches='tight')
                output = climage.convert("fig.png")
                os.system('clear')
                print(output)
            else:
                plt.pause(.001)
                plt.draw()
        if self.mode == "gpu":
            alpha = torch.ones([images.shape[0], images.shape[1], 1], dtype=torch.float32, device=self.device)
            images = torch.cat([images, alpha], dim=-1)
            self.window.draw(images)
        
        if self.mode in ["headless", "debug", "gpu", "gpuless"]:
            return imagesHeadless

    def renderNew(self, simBVH, simModel):
        images = None
        imageRows = []
        sqrtNumEnvs = int(math.sqrt(simModel.numEnvs))

        with wp.ScopedDevice("cuda:0"):

            with wp.ScopedTimer("Render", active=False, detailed=False):

                if self.renderGraph == None:
                    wp.capture_begin()
                    self.renderFunctionNew(simBVH)
                    self.renderGraph = wp.capture_end()
                wp.capture_launch(self.renderGraph)
                image = wp.to_torch(self.pixels).view(self.numEnvs, self.height, self.width, 3)

                for i in range(simModel.numEnvs):

                    if self.mode in ["human", "debug", "terminal", "gpu", "gpudebug"]:
                        rowId = int(i / sqrtNumEnvs)
                        columnId = i % sqrtNumEnvs

                        # arrange in 2D
                        if columnId == 0:
                            imageRows.append(image[i])
                        else:
                            imageRows[rowId] = torch.cat((imageRows[rowId], image[i]), 0)
                
            if self.mode in ["human", "debug", "terminal", "gpu", "gpudebug"]:
                lastRowShape = imageRows[-1].shape
                if not imageRows[0].shape == lastRowShape:
                    missingImages = torch.ones((imageRows[0].shape[0] - lastRowShape[0], lastRowShape[1], lastRowShape[2]), device=self.device)
                    imageRows[-1] = torch.cat((imageRows[-1], missingImages), 0)
                images = torch.cat((imageRows), 1)
        
        if self.mode in ["human", "debug", "terminal"]:
            if self.img is None:
                plt.axis('off')
                self.img = plt.imshow(images.cpu().numpy())
            else:
                self.img.set_data(images.cpu().numpy())

            if self.mode == "terminal":
                plt.savefig("fig.png", bbox_inches='tight')
                output = climage.convert("fig.png")
                os.system('clear')
                print(output)
            else:
                plt.pause(.001)
                plt.draw()

        if self.mode == "gpu":
            alpha = torch.ones([images.shape[0], images.shape[1], 1], dtype=torch.float32, device=wp.device_to_torch(self.device))
            images = torch.cat([images, alpha], dim=-1)
            self.window.draw(images)

        if self.mode == "gpudebug":
            alpha = torch.ones([images.shape[0], images.shape[1], 1], dtype=torch.float32, device=wp.device_to_torch(self.device))
            images = torch.cat([images, alpha], dim=-1)
            return images
        
        if self.mode in ["headless", "debug", "gpu", "gpuless"]:
            return image
            
    def moveCameraForward(self, distance:float=0.1) -> None:
        self.cameraPos += wp.vec3(0.0, 0.0, -distance)
        cameraPosHost = wp.array(self.cameraPos, dtype=wp.vec3, device="cpu")
        wp.copy(self.camera.pos, cameraPosHost)
            
    def moveCameraLeft(self, distance:float=0.1) -> None:
        self.cameraPos += wp.vec3(-distance, 0.0, 0.0)
        cameraPosHost = wp.array(self.cameraPos, dtype=wp.vec3, device="cpu")
        wp.copy(self.camera.pos, cameraPosHost)
            
    def moveCameraUp(self, distance:float=0.1) -> None:
        self.cameraPos += wp.vec3(0.0, distance, 0.0)
        cameraPosHost = wp.array(self.cameraPos, dtype=wp.vec3, device="cpu")
        wp.copy(self.camera.pos, cameraPosHost)

    def rotateCameraHV(self, distanceX:float=5.0, distanceY:float=5.0, scale:float=0.1) -> None:
        self.cameraRot += wp.vec3(distanceX * scale * math.pi/180.0,
                                  0.0,
                                  distanceY * scale * math.pi/180.0)
        # self.camera.rot = wp.quat_rpy(self.cameraRot[2], self.cameraRot[0], self.cameraRot[1])
        cameraRotHost = wp.array(wp.quat_rpy(self.cameraRot[2], self.cameraRot[0], self.cameraRot[1]), dtype=wp.quat, device="cpu")
        wp.copy(self.camera.rot, cameraRotHost)

    def lookAt(self, at):
        localAt = wp.normalize(at - self.camera.pos.numpy()[0])
        pitch = wp.asin(localAt[1])
        yaw = wp.atan2(localAt[0], localAt[2]) - math.pi

        self.camera.at = at
        self.cameraRot[2] = pitch
        self.cameraRot[0] = yaw
        # self.camera.rot = wp.quat_rpy(self.cameraRot[2], self.cameraRot[0], self.cameraRot[1])
        cameraRotHost = wp.array(wp.quat_rpy(self.cameraRot[2], self.cameraRot[0], self.cameraRot[1]), dtype=wp.quat, device="cpu")
        wp.copy(self.camera.rot, cameraRotHost)

class WarpRaycastRenderer:

    def __init__(self, device, simModel, cameraPos=[3.0, 3.0, 3.0], cameraRot=[0.0, 0.0, 0.0], resolution:int=512, mode="human"):

        self.device=device
        self.mode = mode

        self.cameraPos = wp.vec3(cameraPos[0], cameraPos[1], cameraPos[2])
        self.cameraRot = wp.vec3(cameraRot[0], cameraRot[1], cameraRot[2])

        horizontal_aperture = 36.0
        vertical_aperture = 20.25
        aspect = horizontal_aperture / vertical_aperture
        focal_length = 50.0
        self.height = resolution
        self.width = resolution
        # self.width = int(aspect * self.height)
        self.num_pixels = self.width * self.height

        # set anti-aliasing
        self.num_samples = 1

        # set render mode
        self.render_mode = RenderMode.texture

        # construct camera
        self.camera = Camera()
        self.camera.horizontal = horizontal_aperture
        self.camera.vertical = vertical_aperture
        self.camera.aspect = aspect
        self.camera.e = focal_length
        self.camera.tan = vertical_aperture / (2.0 * focal_length)
        self.camera.pos = self.cameraPos
        self.camera.rot = wp.quat_rpy(self.cameraRot[2], self.cameraRot[0], self.cameraRot[1])

        # construct lights
        self.lights = DirectionalLights()
        self.lights.dirs = wp.array(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), dtype=wp.vec3, device=self.device)
        self.lights.intensities = wp.array(np.array([2.0, 0.2]), dtype=float, device=self.device)
        self.lights.num_lights = 2

        # construct rays
        self.rays_width = self.width * pow(2, self.num_samples)
        self.rays_height = self.height * pow(2, self.num_samples)
        self.num_rays = self.rays_width * self.rays_height
        self.rays = wp.zeros(self.num_rays, dtype=wp.vec3, device=self.device)

        # construct pixels
        self.pixels = wp.zeros(self.num_pixels, dtype=wp.vec3, device=self.device)

        # manufacture texture for this asset
        x = np.arange(256.0)
        xx, yy = np.meshgrid(x, x)
        zz = np.zeros_like(xx)
        texture_host = np.stack((xx, yy, zz), axis=2) / 255.0

        # construct texture
        self.texture = wp.array2d(texture_host, dtype=wp.vec3, device=self.device)

        # create plot window
        if self.mode == "human":
            plt.figure(figsize=(10, 10))
            plt.axis("off")
        self.img = None

        # construct meshes
        self.renderMesh = RenderMesh()
        for simEnvironment in simModel.simEnvironments:
            for simMesh in simEnvironment.simMeshes:

                self.mesh = wp.Mesh(
                    points=simMesh.visPoint,
                    indices=simMesh.visFace
                )

                self.renderMesh.id = self.mesh.id
                self.renderMesh.vertices = simMesh.visPoint
                self.renderMesh.indices = simMesh.visFace
                self.renderMesh.texCoords = simMesh.texCoords
                self.renderMesh.texIndices = simMesh.texIndices
                self.normalSums = wp.zeros(simMesh.numVisPoints, dtype=wp.vec3, device=self.device)
                self.renderMesh.vertexNormals = wp.zeros(simMesh.numVisPoints, dtype=wp.vec3, device=self.device)
                # self.renderMesh.pos = simMesh.translationWP
                self.renderMesh.pos = wp.zeros(1, dtype=wp.vec3, device=self.device)
                # self.renderMesh.rot = simMesh.rotationWP
                self.renderMesh.rot = wp.array(np.array([0.0, 0.0, 0.0, 1.0]), dtype=wp.quat, device=self.device)
                self.renderMesh.numTris = len(self.renderMesh.indices)

                # compute vertex normals
                wp.launch(
                    kernel=vertex_normal_sum_kernel,
                    dim=simMesh.numVisFaces,
                    inputs=[self.renderMesh.vertices, self.renderMesh.indices, self.normalSums])
                wp.launch(
                    kernel=normalize_kernel,
                    dim=simMesh.numVisPoints,
                    inputs=[self.normalSums, self.renderMesh.vertexNormals])

    def render(self, simBVH):

        with wp.ScopedDevice("cuda:0"):

            with wp.ScopedTimer("Render", active=True, detailed=False):
            # raycast
            
                wp.launch(
                    kernel=draw_kernel,
                    dim=self.num_rays,
                    inputs=[
                        self.renderMesh,
                        self.camera,
                        self.texture,
                        self.rays_width,
                        self.rays_height,
                        self.rays,
                        self.lights,
                        self.render_mode
                    ])

                # downsample
                wp.launch(
                    kernel=downsample_kernel,
                    dim=self.num_pixels,
                    inputs=[self.rays, self.pixels, self.rays_width, pow(2, self.num_samples)]
                )

        if self.mode == "human":
            if self.img is None:
                self.img = plt.imshow(self.pixels.numpy().reshape((self.height, self.width, 3)))
            else:
                self.img.set_data(self.pixels.numpy().reshape((self.height, self.width, 3)))
            plt.pause(.1)
            plt.draw()
        else:
            return self.pixels.numpy().reshape((self.height, self.width, 3))
            
    def moveCameraForward(self, distance:float=0.1) -> None:
        self.cameraPos += wp.vec3(0.0, 0.0, -distance)
        self.camera.pos = self.cameraPos
            
    def moveCameraLeft(self, distance:float=0.1) -> None:
        self.cameraPos += wp.vec3(-distance, 0.0, 0.0)
        self.camera.pos = self.cameraPos
            
    def moveCameraUp(self, distance:float=0.1) -> None:
        self.cameraPos += wp.vec3(0.0, distance, 0.0)
        self.camera.pos = self.cameraPos

    def rotateCameraHV(self, distanceX:float=5.0, distanceY:float=5.0, scale:float=0.1) -> None:
        self.cameraRot += wp.vec3(distanceX * scale * math.pi/180.0,
                                  0.0,
                                  distanceY * scale * math.pi/180.0)
        self.camera.rot = wp.quat_rpy(self.cameraRot[2], self.cameraRot[0], self.cameraRot[1])

    def lookAt(self, at):
        localAt = wp.normalize(at - self.camera.pos)
        pitch = wp.asin(localAt[1])
        yaw = wp.atan2(localAt[0], localAt[2]) - math.pi

        self.camera.at = at
        self.cameraRot[2] = pitch
        self.cameraRot[0] = yaw
        self.camera.rot = wp.quat_rpy(self.cameraRot[2], self.cameraRot[0], self.cameraRot[1])
