from setuptools import setup


setup(
    app=["run_app.py"],
    data_files=[
        (
            "ThirdPartyNotices",
            [
                "LICENSE",
                "THIRD_PARTY_NOTICES.md",
                "third_party/GazeMap/LICENSE",
                "third_party/GazeMap/NOTICE",
            ],
        )
    ],
    options={
        "py2app": {
            "argv_emulation": False,
            "packages": ["nosee", "cv2", "mediapipe", "torch", "scipy", "numpy"],
            "includes": ["Quartz", "Security", "Foundation", "AppKit"],
            "plist": {
                "LSUIElement": True,
                "NSCameraUsageDescription": (
                    "NoSee uses the camera locally to detect whether another person "
                    "is looking at this screen. Frames are not saved or uploaded."
                ),
            },
        }
    },
    setup_requires=["py2app"],
)
