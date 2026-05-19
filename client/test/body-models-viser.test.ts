import assert from "node:assert/strict";
import test from "node:test";
import {
  add_body_model,
  addBodyModel,
  BodyModelScene,
  createBodyModel,
  installBodyModelsViserPlugin,
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
  assert.deepEqual(handle.body_pose, [2]);
  assert.equal(forwardCount, 2);
  assert.deepEqual(handle.mesh.bones[0]?.position, [2, 0, 0]);

  handle.shape = [5];
  assert.deepEqual(handle.mesh.vertices, [[5, 0, 0]]);

  handle.set_pose({ global_translation: [3, 0, 0] } as Partial<Record<keyof DemoParams, NumericArray>>);
  assert.deepEqual(handle.mesh.bones[0]?.position, [5, 0, 0]);

  handle.remove();
  assert.deepEqual(calls.slice(-2), ["remove-mesh", "remove-frame"]);
});

test("snake_case export and monkey patch match body_models.extras.viser_plugin shape", () => {
  const scene = new DemoScene();
  const model = demoModel();
  const direct = add_body_model(scene, "/direct", model, { color: [1, 2, 3] });
  assert.equal(direct.name, "/direct");
  direct.remove();

  class SceneApi extends DemoScene {}
  const viser = { SceneApi };
  installBodyModelsViserPlugin(viser);
  const patched = new SceneApi() as SceneApi & {
    add_body_model: typeof add_body_model;
    addBodyModel: typeof addBodyModel;
  };

  const handle = patched.add_body_model("/patched", model);
  assert.equal(handle.name, "/patched");
  assert.equal(patched.add_body_model, patched.addBodyModel);

  class GlobalSceneApi extends DemoScene {}
  installBodyModelsViserPlugin({ viser: { SceneApi: GlobalSceneApi } });
  const globalPatched = new GlobalSceneApi() as GlobalSceneApi & { add_body_model: typeof add_body_model };
  assert.equal(globalPatched.add_body_model("/global", model).name, "/global");
});

function demoModel() {
  return createBodyModel<DemoParams>({
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
      const tx = params.global_translation[0] + params.body_pose[0];
      return {
        skeleton: [transform(tx)],
        mesh: [[params.shape[0], 0, 0]],
      };
    },
  });
}

class DemoScene implements BodyModelScene {
  calls: string[] = [];

  addFrame(name: string, options: { showAxes: boolean }): ViserFrameHandle {
    this.calls.push(`frame:${name}:${options.showAxes}`);
    return {
      name,
      wxyz: [1, 0, 0, 0],
      position: [0, 0, 0],
      visible: true,
      remove: () => this.calls.push("remove-frame"),
    };
  }

  addMeshSkinned(name: string, options: Parameters<BodyModelScene["addMeshSkinned"]>[1]): ViserSkinnedMeshHandle {
    this.calls.push(`mesh:${name}:${options.faces.length}:${options.vertices.length}`);
    return {
      vertices: options.vertices,
      bones: [{ wxyz: [1, 0, 0, 0], position: [0, 0, 0] }],
      remove: () => this.calls.push("remove-mesh"),
    };
  }
}
