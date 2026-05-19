import assert from "node:assert/strict";
import test from "node:test";
import {
  addBodyModel,
  BodyModelScene,
  createBodyModel,
  Mat4,
  NumericArray,
  Vec3,
  ViserBoneHandle,
  ViserFrameHandle,
  ViserSkinnedMeshHandle,
} from "../src/index";

interface DemoParams {
  shape: number[];
  body_pose: number[];
  global_translation: Vec3;
}

function transform(tx: number): Mat4 {
  return [
    [1, 0, 0, tx],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

test("addBodyModel mirrors the Python viser plugin handle shape", () => {
  const calls: string[] = [];
  const bones: ViserBoneHandle[] = [{ wxyz: [1, 0, 0, 0], position: [0, 0, 0] }];
  const scene: BodyModelScene = {
    addFrame(name, options): ViserFrameHandle {
      calls.push(`frame:${name}:${options.showAxes}`);
      return {
        name,
        wxyz: [1, 0, 0, 0],
        position: [0, 0, 0],
        visible: true,
        remove: () => calls.push("remove-frame"),
      };
    },
    addMeshSkinned(name, options): ViserSkinnedMeshHandle {
      calls.push(`mesh:${name}:${options.faces.length}:${options.vertices.length}`);
      assert.deepEqual(options.skinWeights[0], [0.7, 0.3]);
      return {
        vertices: options.vertices,
        bones,
        remove: () => calls.push("remove-mesh"),
      };
    },
  };

  let forwardCount = 0;
  const model = createBodyModel<DemoParams>({
    modelName: "Demo",
    faces: [[0, 1, 2, 3]],
    skinWeights: [[0.7, 0.3]],
    poseParameterNames: ["body_pose", "global_translation"],
    getRestPose: () => ({
      shape: [0],
      body_pose: [0],
      global_translation: [0, 0, 0],
    }),
    forward: (params) => {
      forwardCount += 1;
      const tx = params.global_translation[0] + params.body_pose[0];
      return {
        skeleton: [transform(tx)],
        mesh: [[params.shape[0], 0, 0]],
      };
    },
  });

  const handle = addBodyModel(scene, "/body", model);
  assert.equal(handle.name, "/body");
  assert.deepEqual(calls, ["frame:/body:false", "mesh:/body/mesh:2:1"]);

  handle.bodyPose = [2];
  assert.equal(forwardCount, 2);
  assert.deepEqual(handle.mesh.bones[0]?.position, [2, 0, 0]);

  handle.shape = [5];
  assert.deepEqual(handle.mesh.vertices, [[5, 0, 0]]);

  handle.setPose({ global_translation: [3, 0, 0] } as Partial<Record<keyof DemoParams, NumericArray>>);
  assert.deepEqual(handle.mesh.bones[0]?.position, [5, 0, 0]);

  handle.remove();
  assert.deepEqual(calls.slice(-2), ["remove-mesh", "remove-frame"]);
});
