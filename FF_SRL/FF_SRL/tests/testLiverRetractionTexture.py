import torch
from pynput import keyboard

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np
import FF_SRL as dk

wp.init()

try:
    from pxr import Usd
except ModuleNotFoundError:
    print("No pxr package")

class TestStandalone():
    def __init__(self, numEnvs=1, numAct=3, debugTime=True, useGraph=True, visualize=True) -> None:

        self.device = "cuda:0"
        self.visualize = visualize
        self.displayTiming = debugTime
        self.useGraph = useGraph
        self.capturedGraph = None
        self.renderGraph = None

        self.workspaceHigh = torch.tensor([7.0, 9.0, 7.4], dtype=torch.float, device=self.device)
        self.workspaceLow = torch.tensor([-7.0, 0.7, -10.6], dtype=torch.float, device=self.device)
        
        self.actionStrength = torch.tensor((0.1, 0.1, 0.1), dtype = torch.float, device = self.device)

        self.numEnvs = numEnvs
        self.numActs = numAct

        self.fps = 20
        self.dt = 1. / float(self.fps)
        self.sim_substeps = 20
        
        self.constraintSteps = 1
        self.sim_dt = self.dt / float(self.sim_substeps)
        self.sim_time = 0.0
        self.render_time = 0.0

        self.stage = Usd.Stage.Open("../scenes/liverRetractionTexture.usd")
        self.simModel = dk.SimModelDO(self.stage, self.numEnvs, self.device, globalKsDrag=1.0, simFrameRate=self.fps, simConstraintsSteps=self.constraintSteps, simSubsteps=self.sim_substeps, globalLaparoscopeDragLookupRadius=1.0, globalKsDistance=1., globalKsVolume=1.)
        self.simBVH = dk.SimBVH(self.device, self.simModel, binsNumber=8, load=True, save=False, path=("../scenes/liverRetractionTexture.npz"))

        self.renderer = dk.render.WarpRaycastRendererDO(device=self.device,
                                                        simModel=self.simModel,
                                                        resolution=256,
                                                        cameraPos=[2.0, 22.0, 40.0],
                                                        cameraRot=[0.0, 0.0, 0.0],
                                                        lightPos=[0.0, 15.0, 80.0],
                                                        lightIntensity=0.03,
                                                        mode="gpu",
                                                        horizontalAperture=45,
                                                        verticalAperture=45)
        # self.renderer.lookAt([0.0, 10, 0.0])
        self.renderer.lookAt([-2.75, 22.5, -4.0])

        self.simIntegrator = dk.SimIntegratorDO(self.device)

        self.remoteCenterOfMotionPos = torch.tensor([9.036183, 40.260103, 5.567807], dtype = torch.float, device = self.device).repeat(self.numEnvs, 1)
        self.remoteCenterOfMotionRot = torch.tensor([0.0, 0.0, 0.0], dtype = torch.float, device = self.device).repeat(self.numEnvs, 1)
        self.simModel.setRotationCenter(wp.from_torch(self.remoteCenterOfMotionPos, dtype=wp.vec3), wp.from_torch(self.remoteCenterOfMotionRot, dtype=wp.vec3))

        # Set the position of the effector
        self.effectorPosition = torch.tensor([0.0, 23.0, -2.0], dtype = torch.float, device = self.device)
        # self.effectorPosition = torch.tensor([0.0, 10.0, 4.0], dtype = torch.float, device = self.device)
        currentEffectorPosition = self.simModel.getLaparoscopePositionsTensor()
        shift = self.effectorPosition - currentEffectorPosition
        self.simModel.applyCartesianActions(wp.from_torch(shift[0], dtype=wp.float32))

    def step(self, actions=None, actionsCartesian=None):
        self.simModel.resetCollisionInfo()

        if not actions is None:
            with wp.ScopedTimer("ApplyActions", active=self.displayTiming, detailed=False):
                self.simModel.applyActions(actions)
                if actionsCartesian:
                    self.simModel.applyCartesianActionsInWorkspace(actionsCartesian)

        elif not actionsCartesian is None:
            self.simModel.applyCartesianActionsInWorkspace(actionsCartesian)

        if self.useGraph == True and self.capturedGraph == None:
            with wp.ScopedTimer("Capture graph", active=self.displayTiming, detailed=False):
                wp.capture_begin()
                if testStandalone.simModel.reduce == True:
                    self.simIntegrator.stepModelReduce(self.simModel)
                else:
                    self.simIntegrator.stepModel(self.simModel)
                self.capturedGraph = wp.capture_end()

        with wp.ScopedTimer("Simulate step", active=self.displayTiming, detailed=False):
            if self.useGraph == True:
                wp.capture_launch(self.capturedGraph)
            else:
                if testStandalone.simModel.reduce == True:
                    self.simIntegrator.stepModelReduce(self.simModel)
                else:
                    self.simIntegrator.stepModel(self.simModel)
            self.sim_time += self.dt

    def render(self):

        if self.renderGraph == None:
            wp.capture_begin()
            self.renderer.renderNew(self.simBVH)
            self.renderGraph = wp.capture_end()
        else:
            wp.capture_launch(self.renderGraph)

numAct = 3
nEnvs = 1
maxIter = 10000

if len(sys.argv) > 1:
    nEnvs = int(sys.argv[1])

testStandalone = TestStandalone(numEnvs=nEnvs, debugTime=False, useGraph=True, visualize=True)

exitScript = False
clamp = 0.0

keyboardActions = [0] * numAct

def on_press(key):
    global exitScript
    global clamp
    try:
        if key.char == "u":
            keyboardActions[1] = 1.0
        if key.char == "o":
            keyboardActions[1] = -1.0
        if key.char == "i":
            keyboardActions[2] = -1.0
        if key.char == "k":
            keyboardActions[2] = 1.0
        if key.char == "j":
            keyboardActions[0] = -1.0
        if key.char == "l":
            keyboardActions[0] = 1.0
        if key.char == "f":
            envs = torch.tensor([1] * testStandalone.numEnvs, dtype=torch.int32, device=testStandalone.device)
            testStandalone.simModel.forceLaparoscopeClamp(envs)
        if key.char == "r":
            envs = list(range(0, testStandalone.numEnvs))
            testStandalone.simModel.resetFixedModel(envs)

    except AttributeError:
        if key == keyboard.Key.esc:
            print("exiting")
            exitScript = True

input = True

if input:
    keyboardListener = keyboard.Listener(on_press=on_press)
    keyboardListener.start()

for i in range(maxIter):
    actions = torch.zeros((testStandalone.numEnvs - 1, numAct), device = 'cuda:0')
    actionsKey = torch.tensor(keyboardActions, dtype=torch.float32, device = 'cuda:0')
    keyboardActions = [0] * numAct
    actionsKey = torch.clip(actionsKey, -1., 1.) * testStandalone.actionStrength
    actions = actionsKey.repeat(testStandalone.numEnvs)

    testStandalone.step(actionsCartesian=wp.from_torch(actions))
    testStandalone.simBVH.refitBVH(useGraph=True)
    image = testStandalone.renderer.renderNew(testStandalone.simBVH, testStandalone.simModel)

    if exitScript:
        break

if input:
    keyboardListener.stop()