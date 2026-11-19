# Third-party notices

NoSee uses or downloads the following third-party components and model assets.

## GazeMap

- Project: https://github.com/fammad/GazeMap
- Code license: Apache License 2.0. The upstream project requests that its
  `NOTICE` file and author credit be retained when its code is redistributed.
- Pretrained weights: CC BY-NC-SA 4.0; non-commercial use and redistribution
  only, with attribution and share-alike terms.
- Model source: MPIIGaze, whose dataset/model terms are non-commercial.
- The model is a research/demo estimator and is not independently validated
  for detecting bystanders.

## OpenCV Zoo SFace

- Project: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
- Model: `face_recognition_sface_2021dec.onnx`
- SHA-256: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`
- The upstream model directory states that its files are licensed under
  Apache License 2.0. The model installer stores the upstream license text
  next to the downloaded model when available.

## MediaPipe Face Landmarker

- Project: https://github.com/google-ai-edge/mediapipe
- Model download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- The task file is downloaded by the user at setup time and remains on the
  device. Consult the upstream project and model-card terms before
  redistribution.

## PyObjC, OpenCV, PyTorch, SciPy, NumPy

These are installed as Python dependencies, not copied into the source tree.
Their licenses and notices are available from their respective upstream
projects and package distributions.
