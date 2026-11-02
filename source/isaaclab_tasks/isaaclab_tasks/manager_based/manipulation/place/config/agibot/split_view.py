# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Split viewport setup for Agibot Toy2Box teleoperation."""

import asyncio
from pathlib import Path

import omni.kit.app
import omni.ui as ui
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from omni.kit.viewport.utility import create_viewport_window, get_active_viewport_window
from pxr import Gf, UsdGeom


async def _dock_top_view(top_window, side_dock_window):
    """Dock after the newly created viewport has been registered by the UI."""
    top_window.deferred_dock_in(side_dock_window.title)
    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()
    top_window.dock_in(side_dock_window, ui.DockPosition.LEFT, 0.5)


def _create_controls_overlay(side_window):
    """Show keyboard controls over the side viewport without changing its size."""
    # Outlined SVG text keeps Korean readable without changing Kit's global fonts.
    controls_image = Path(__file__).with_name("keyboard_controls.svg")
    frame = side_window.get_frame("agibot_keyboard_controls")
    frame.clear()
    frame.opaque_for_mouse_events = False
    with frame:
        with ui.VStack(opaque_for_mouse_events=False):
            ui.Spacer(height=56)
            with ui.HStack(height=332, opaque_for_mouse_events=False):
                ui.Spacer()
                ui.Image(
                    str(controls_image),
                    width=300,
                    height=332,
                    fill_policy=ui.FillPolicy.PRESERVE_ASPECT_FIT,
                    opaque_for_mouse_events=False,
                )
                ui.Spacer(width=16)
            ui.Spacer()


def create_split_view():
    """Create equally sized top and side viewports for the current stage."""
    stage = get_current_stage()
    top_camera_path = "/World/AgibotTopViewCamera"
    side_camera_path = "/World/AgibotSideViewCamera"

    top_camera = UsdGeom.Camera.Define(stage, top_camera_path)
    UsdGeom.Camera.Define(stage, side_camera_path)

    side_window = get_active_viewport_window()
    if side_window is None:
        raise RuntimeError("No active viewport is available for split view.")
    side_dock_window = ui.Workspace.get_window(side_window.title)
    if side_dock_window is None:
        raise RuntimeError(f"Could not find the '{side_window.title}' window in the UI workspace.")

    top_window = create_viewport_window(name="Top View")
    top_window.viewport_api.camera_path = top_camera_path
    side_window.viewport_api.camera_path = side_camera_path
    _create_controls_overlay(side_window)

    target = (0.5, 0.0, 0.0)
    top_camera_xform = UsdGeom.XformCommonAPI(top_camera)
    top_camera_xform.SetTranslate(Gf.Vec3d(0.1, -0.45, 3.0))
    top_camera_xform.SetRotate(Gf.Vec3f(0.0, 0.0, -90.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    set_camera_view((0.5, -3.0, 0.5), target, side_camera_path, side_window.viewport_api)

    dock_task = asyncio.ensure_future(_dock_top_view(top_window, side_dock_window))
    for window_name in (
        "Stage",
        "Layer",
        "Render Settings",
        "Property",
        "Content",
        "Console",
        "IsaacLab",
        "Semantics Schema Editor",
    ):
        window = ui.Workspace.get_window(window_name)
        if window is not None:
            window.visible = False
    return top_window, side_window, dock_task
