import assert from "node:assert/strict";
import test from "node:test";
import { skinVertices, Mat4 } from "../src/index";

const identity: Mat4 = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

const translateX: Mat4 = [
  [1, 0, 0, 10],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

const translateY: Mat4 = [
  [1, 0, 0, 0],
  [0, 1, 0, 20],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

test("skinVertices uses every provided weight", () => {
  assert.deepEqual(
    skinVertices({
      vertices: [[1, 2, 3]],
      skinWeights: [[0.2, 0.3, 0.5]],
      skinJoints: [[0, 1, 2]],
      boneTransforms: [identity, translateX, translateY],
    }),
    [[4, 12, 3]],
  );
});

test("skinVertices rejects malformed weights", () => {
  assert.throws(
    () =>
      skinVertices({
        vertices: [[1, 2, 3]],
        skinWeights: [[1]],
        skinJoints: [[0, 1]],
        boneTransforms: [identity],
      }),
    /Invalid skinning weights/,
  );
});
