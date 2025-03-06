"""
https://github.com/tensorflow/tfjs-models/blob/838611c02f51159afdd77469ce67f0e26b7bbb23/face-landmarks-detection/src/mediapipe-facemesh/keypoints.ts

 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 """

mesh_annotations = {
  # ---------------------------------------------------------------------------------------------------------
  # These are the selected features
  # ---------------------------------------------------------------------------------------------------------
  "rightEyebrowFull": [156, 70, 63, 105, 66, 107, 55, 193, 35, 124, 46, 53, 52, 65],
  "leftEyebrowFull": [383, 300, 293, 334, 296, 336, 285, 417, 265, 353, 276, 283, 282, 295],

  "rightInnerCanthus": [173],
  "leftInnerCanthus": [398],

  "leftLipCorners": [291, 308, 409, 375],
  "rightLipCorners": [61, 78, 185, 146],
  "upperLipMidpoint": [37, 267],
  "centreUpperLipMidpoint": [0],
  "leftInnerCanthusAndUpperLipMidpoint": [398, 267],
  "rightEyeBox": [356],
  "leftEyeBox": [127],
  "bottomOfChin": [152],
  "leftFace": [127],
  "rightFace": [356],
  "topFace": [10],
  # ---------------------------------------------------------------------------------------------------------
  # Second Attempt
  # ---------------------------------------------------------------------------------------------------------
  0: [70], 1: [63], 2: [105], 3: [66], 4: [55], # right eyebrow 0-4
  5: [285], 6: [296], 7: [334], 8: [293], 9: [300], # left eyebrow 5-9
  10: [130], 11: [160], 12: [158], 13: [173], # upper right eye 10-13
  14: [153], 15: [144], # lower right eye 14-15
  16: [398], 17: [385], 18: [387], 19: [359], # upper left eye 16-19
  20: [373], 21: [380], # lower left eye 20-21
  22: [1], # nose centre
  23: [98], 27: [327], # nose
  28: [61], 29: [39], 30: [37], 31: [0], 32: [267], 33: [269], 34: [291], # upper outer lip
  35: [405], 36: [314], 37: [17], 38: [84], 39: [181], # lower outer lip
  40: [191], 41: [81], 42: [13], 43: [311], 44: [415], # upper inner lip
  45: [402], 46: [14], 47: [178], # lower inner lip
  48: [127], 49: [356], # side of face
  50: [152] # bottom of chin
}
selected_features = []

for key in mesh_annotations.keys():
  selected_features += mesh_annotations[key]
selected_features = set(selected_features) # This is to ensure coordinates used in different features are not repeated