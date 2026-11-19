from __future__ import annotations

import threading
from pathlib import Path

import objc
from AppKit import (
    NSAlert,
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSColor,
    NSFont,
    NSMenu,
    NSMenuItem,
    NSMakePoint,
    NSMakeRect,
    NSOpenPanel,
    NSNonactivatingPanelMask,
    NSPanel,
    NSStatusBar,
    NSStatusWindowLevel,
    NSVariableStatusItemLength,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskTitled,
    NSWindowStyleMaskBorderless,
    NSButton,
    NSTextField,
    NSOperationQueue,
)
from Foundation import NSObject

from .install_models import install_models
from .monitor import MonitorController, MonitorUpdate
from .paths import FACE_MODEL, LANDMARKER_MODEL, SFACE_MODEL, GAZEMAP_DIR
from .storage import LocalStore

WARNING_TEXT = "当心隐私信息泄漏，有人偷看"


class AppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        self.store = LocalStore()
        self.monitor = MonitorController(self._monitor_update, self.store)
        self._installing = False
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        button = self.status_item.button()
        button.setTitle_("NoSee")
        self._build_menu()
        self._build_main_window()
        self._build_warning_panel()
        self._set_status("监控已停止")
        self._refresh_controls()
        self.window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return False

    @objc.python_method
    def _build_menu(self):
        menu = NSMenu.alloc().init()
        self.menu_status_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "监控已停止", None, ""
        )
        self.menu_status_item.setEnabled_(False)
        menu.addItem_(self.menu_status_item)
        menu.addItem_(NSMenuItem.separatorItem())
        self.open_item = self._menu_item(menu, "打开控制面板", "openPanel:")
        menu.addItem_(NSMenuItem.separatorItem())
        self.start_item = self._menu_item(menu, "开始监控", "startMonitoring:")
        self.pause_item = self._menu_item(menu, "暂停监控", "pauseMonitoring:")
        self.enroll_item = self._menu_item(menu, "录入本人", "enrollOwner:")
        self.clear_item = self._menu_item(menu, "清除本人资料…", "clearOwner:")
        self.install_item = self._menu_item(menu, "安装离线模型…", "installModels:")
        menu.addItem_(NSMenuItem.separatorItem())
        self._menu_item(menu, "退出 NoSee", "quitApp:")
        self.status_item.setMenu_(menu)

    @objc.python_method
    def _menu_item(self, menu, title, action):
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
        item.setTarget_(self)
        menu.addItem_(item)
        return item

    @objc.python_method
    def _label(self, parent, text, frame, size=14, color=None, bold=False):
        label = NSTextField.labelWithString_(text)
        label.setFrame_(frame)
        font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
        label.setFont_(font)
        if color is not None:
            label.setTextColor_(color)
        parent.addSubview_(label)
        return label

    @objc.python_method
    def _button(self, parent, title, action, frame):
        button = NSButton.alloc().initWithFrame_(frame)
        button.setTitle_(title)
        button.setTarget_(self)
        button.setAction_(action)
        parent.addSubview_(button)
        return button

    @objc.python_method
    def _build_main_window(self):
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 480, 310),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("NoSee 本地防窥提醒")
        content = self.window.contentView()
        self._label(content, "NoSee", NSMakeRect(24, 258, 420, 30), size=22, bold=True)
        self.status_label = self._label(content, "监控已停止", NSMakeRect(24, 225, 430, 24), size=14, bold=True)
        self._label(
            content,
            "摄像头画面只在本机内存中处理，不保存、不上传。\n"
            "普通摄像头只能估算视线；这是一项提醒功能，不能保证拦截所有旁观。",
            NSMakeRect(24, 165, 430, 52),
            size=12,
            color=NSColor.secondaryLabelColor(),
        )
        self._button(content, "录入本人", "enrollOwner:", NSMakeRect(24, 108, 128, 34))
        self._button(content, "开始监控", "startMonitoring:", NSMakeRect(166, 108, 128, 34))
        self._button(content, "暂停监控", "pauseMonitoring:", NSMakeRect(308, 108, 128, 34))
        self._button(content, "安装离线模型", "installModels:", NSMakeRect(24, 58, 150, 30))
        self._button(content, "清除本人资料", "clearOwner:", NSMakeRect(190, 58, 150, 30))
        self._label(
            content,
            "模型和本人特征保存在此 Mac。开始前请在系统设置中允许摄像头访问。",
            NSMakeRect(24, 20, 430, 28),
            size=11,
            color=NSColor.secondaryLabelColor(),
        )
        self.window.center()

    @objc.python_method
    def _build_warning_panel(self):
        style = NSWindowStyleMaskBorderless | NSNonactivatingPanelMask
        self.warning_panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 540, 84), style, NSBackingStoreBuffered, False
        )
        self.warning_panel.setLevel_(NSStatusWindowLevel)
        self.warning_panel.setOpaque_(False)
        self.warning_panel.setHasShadow_(True)
        self.warning_panel.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(
            0.73, 0.08, 0.05, 0.96
        ))
        self.warning_panel.setIgnoresMouseEvents_(True)
        self.warning_panel.setHidesOnDeactivate_(False)
        self.warning_panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorFullScreenAuxiliary
            | NSWindowCollectionBehaviorStationary
        )
        label = NSTextField.labelWithString_(WARNING_TEXT)
        label.setFrame_(NSMakeRect(18, 23, 504, 40))
        label.setFont_(NSFont.boldSystemFontOfSize_(22))
        label.setTextColor_(NSColor.whiteColor())
        label.setAlignment_(1)  # center
        self.warning_panel.contentView().addSubview_(label)

    @objc.python_method
    def _set_status(
        self,
        text: str,
        warning: bool = False,
        other_faces: int = 0,
        screen_facing_faces: int = 0,
        unknown_gaze: bool = False,
        uncertain_identity_faces: int = 0,
    ):
        if hasattr(self, "status_label"):
            self.status_label.setStringValue_(text)
        if hasattr(self, "status_item"):
            active = text.startswith("监控中：")
            if warning:
                title = "⚠ 有人看屏幕"
            elif active and uncertain_identity_faces and other_faces:
                title = f"NoSee · 旁人{other_faces}/待确认{uncertain_identity_faces}"
            elif active and uncertain_identity_faces:
                title = f"NoSee · 身份确认中{uncertain_identity_faces}"
            elif active and unknown_gaze:
                title = f"NoSee · 旁人{other_faces} 视线未知"
            elif active and other_faces:
                title = f"NoSee · 旁人{other_faces}/朝屏{screen_facing_faces}"
            else:
                title = "NoSee · 监控中" if active else "NoSee"
            self.status_item.button().setTitle_(title)
        if hasattr(self, "menu_status_item"):
            title = f"⚠ {text}" if warning else text
            self.menu_status_item.setTitle_(title)

    @objc.python_method
    def _refresh_controls(self):
        if not hasattr(self, "start_item"):
            return
        running = self.monitor.running
        busy = self.monitor.busy or self._installing
        self.start_item.setEnabled_(not running and not busy)
        self.pause_item.setEnabled_(running or busy)
        self.enroll_item.setEnabled_(not running and not busy)
        self.install_item.setEnabled_(not running and not busy)
        self.clear_item.setEnabled_(not running and not busy)

    @objc.python_method
    def _monitor_update(self, update: MonitorUpdate):
        NSOperationQueue.mainQueue().addOperationWithBlock_(lambda: self._apply_update(update))

    @objc.python_method
    def _apply_update(self, update: MonitorUpdate):
        self._set_status(
            update.status,
            update.warning,
            update.other_faces,
            update.screen_facing_faces,
            update.unknown_gaze,
            update.uncertain_identity_faces,
        )
        if update.warning:
            screen = __import__("AppKit").NSScreen.mainScreen()
            frame = screen.visibleFrame()
            size = self.warning_panel.frame().size
            x = frame.origin.x + (frame.size.width - size.width) / 2
            y = frame.origin.y + frame.size.height - size.height - 28
            self.warning_panel.setFrameOrigin_(NSMakePoint(x, y))
            self.warning_panel.orderFrontRegardless()
        else:
            self.warning_panel.orderOut_(None)
        self._refresh_controls()

    def openPanel_(self, sender):
        self.window.makeKeyAndOrderFront_(None)
        NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def startMonitoring_(self, sender):
        self.monitor.start_monitoring()
        self._refresh_controls()

    def pauseMonitoring_(self, sender):
        self.monitor.stop_monitoring()
        self._refresh_controls()

    def enrollOwner_(self, sender):
        self.monitor.enroll_owner()
        self._refresh_controls()

    def clearOwner_(self, sender):
        if self.monitor.running or self.monitor.busy:
            self._apply_update(MonitorUpdate("请先暂停监控或等待当前操作结束", False))
            return
        alert = NSAlert.alloc().init()
        alert.setMessageText_("清除本人脸部资料？")
        alert.setInformativeText_("这会从此 Mac 的钥匙圈中删除已录入的特征向量。原始画面没有保存。")
        alert.addButtonWithTitle_("清除")
        alert.addButtonWithTitle_("取消")
        if alert.runModal() == 1000:
            self.monitor.clear_owner()
        self._refresh_controls()

    def installModels_(self, sender):
        if self.monitor.running or self.monitor.busy or self._installing:
            self._apply_update(MonitorUpdate("请先暂停监控或等待当前操作结束", False))
            return
        face_model_source = None
        if not FACE_MODEL.is_file():
            panel = NSOpenPanel.openPanel()
            panel.setCanChooseFiles_(True)
            panel.setCanChooseDirectories_(False)
            panel.setAllowsMultipleSelection_(False)
            panel.setAllowedFileTypes_(["mat"])
            panel.setTitle_("选择 MPIIGaze 六点人脸模型")
            panel.setMessage_(
                "GazeMap 需要 MPIIGaze 数据集中的“6 points-based face model.mat”。"
                "请先从官方数据集解压该文件，再在此选择。"
            )
            panel.setPrompt_("选择文件")
            if panel.runModal() != 1:
                return
            face_model_source = Path(str(panel.URL().path()))
        self._installing = True
        self._set_status("正在下载离线模型；只下载模型，不会上传摄像头画面…")
        self._refresh_controls()

        def work():
            try:
                install_models(face_model_source)
                message = "离线模型安装完成。现在可以录入本人。"
            except Exception as exc:
                message = f"模型安装失败：{exc}"
            update = MonitorUpdate(message, False)

            def finished():
                self._installing = False
                self._apply_update(update)
                self._refresh_controls()

            NSOperationQueue.mainQueue().addOperationWithBlock_(finished)

        threading.Thread(target=work, name="nosee-model-installer", daemon=True).start()

    def quitApp_(self, sender):
        self.monitor.stop_monitoring()
        NSApplication.sharedApplication().terminate_(None)


def main() -> None:
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()

