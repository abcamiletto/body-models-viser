export type {
  BodyModelForwardOutput,
  BodyModelLike,
  BodyModelParams,
  BodyModelScene,
  Face,
  Mat4,
  NumericArray,
  QuatWxyz,
  Vec3,
  ViserBoneHandle,
  ViserFrameHandle,
  ViserSkinnedMeshHandle,
} from "./generatedTypes";

import type {
  BodyModelForwardOutput,
  BodyModelLike,
  BodyModelParams,
  BodyModelScene,
  Mat4,
  NumericArray,
  QuatWxyz,
  Vec3,
  ViserFrameHandle,
  ViserSkinnedMeshHandle,
} from "./generatedTypes";

export class ViserBodyModelHandle<TParams extends BodyModelParams = BodyModelParams> {
  readonly modelName: string;

  constructor(
    readonly model: BodyModelLike<TParams>,
    readonly pose: TParams,
    readonly rootFrame: ViserFrameHandle,
    readonly mesh: ViserSkinnedMeshHandle,
  ) {
    this.modelName = model.modelName;
  }

  get name(): string {
    return this.rootFrame.name;
  }

  get wxyz(): QuatWxyz {
    return this.rootFrame.wxyz;
  }

  set wxyz(value: QuatWxyz) {
    this.rootFrame.wxyz = value;
  }

  get position(): Vec3 {
    return this.rootFrame.position;
  }

  set position(value: Vec3) {
    this.rootFrame.position = value;
  }

  get visible(): boolean {
    return this.rootFrame.visible;
  }

  set visible(value: boolean) {
    this.rootFrame.visible = value;
  }

  get shape(): NumericArray {
    return this.param("shape");
  }

  set shape(value: NumericArray) {
    this.setPoseParam("shape", value);
  }

  get bodyPose(): NumericArray {
    return this.param("body_pose");
  }

  set bodyPose(value: NumericArray) {
    this.setPoseParam("body_pose", value);
  }

  get handPose(): NumericArray {
    return this.param("hand_pose");
  }

  set handPose(value: NumericArray) {
    this.setPoseParam("hand_pose", value);
  }

  get headPose(): NumericArray {
    return this.param("head_pose");
  }

  set headPose(value: NumericArray) {
    this.setPoseParam("head_pose", value);
  }

  get expression(): NumericArray {
    return this.param("expression");
  }

  set expression(value: NumericArray) {
    this.setPoseParam("expression", value);
  }

  get globalRotation(): NumericArray {
    return this.param("global_rotation");
  }

  set globalRotation(value: NumericArray) {
    this.setPoseParam("global_rotation", value);
  }

  get globalTranslation(): NumericArray {
    return this.param("global_translation");
  }

  set globalTranslation(value: NumericArray) {
    this.setPoseParam("global_translation", value);
  }

  setPose(updates: Partial<TParams>): void {
    let changed = false;
    let rebuildMesh = false;
    for (const [name, value] of Object.entries(updates) as [keyof TParams & string, NumericArray][]) {
      const current = this.param(name);
      if (numericArrayEqual(current, value)) {
        continue;
      }
      this.pose[name] = cloneNumericArray(value) as TParams[keyof TParams & string];
      changed = true;
      rebuildMesh ||= !this.model.poseParameterNames.includes(name);
    }
    if (changed) {
      this.applyPose({ rebuildMesh });
    }
  }

  remove(): void {
    this.mesh.remove();
    this.rootFrame.remove();
  }

  private param(name: string): NumericArray {
    if (!(name in this.pose)) {
      throw new Error(`${this.modelName} does not support ${JSON.stringify(name)}.`);
    }
    return this.pose[name]!;
  }

  private setPoseParam(name: string, value: NumericArray): void {
    this.setPose({ [name]: value } as Partial<TParams>);
  }

  private applyPose({ rebuildMesh }: { rebuildMesh: boolean }): void {
    if (rebuildMesh) {
      const bindParams = this.model.getBindParams(this.pose);
      const bindOutput = this.model.forward(bindParams);
      this.mesh.vertices = [...bindOutput.mesh];
    }
    const skeleton = this.model.forwardSkeleton(this.pose);
    const { boneWxyzs, bonePositions } = bonePoses(skeleton);
    if (boneWxyzs.length !== this.mesh.bones.length) {
      throw new Error(
        `${this.modelName} produced ${boneWxyzs.length} skeleton joints for ${this.mesh.bones.length} mesh bones.`,
      );
    }
    for (let i = 0; i < this.mesh.bones.length; i++) {
      this.mesh.bones[i]!.wxyz = boneWxyzs[i]!;
      this.mesh.bones[i]!.position = bonePositions[i]!;
    }
  }
}

export function addBodyModel<TParams extends BodyModelParams>(
  scene: BodyModelScene,
  name: string,
  model: BodyModelLike<TParams>,
  options: { color?: [number, number, number] } = {},
): ViserBodyModelHandle<TParams> {
  if (model.isRigidBody) {
    throw new Error("addBodyModel() only supports non-rigid models.");
  }
  const pose = model.getRestPose();
  const bindPose = model.getBindParams(pose);
  const bindOutput = model.forward(bindPose);
  const { boneWxyzs, bonePositions } = bonePoses(bindOutput.skeleton);
  const root = scene.addFrame(name, { showAxes: false });
  const mesh = scene.addMeshSkinned(`${name}/mesh`, {
    vertices: [...bindOutput.mesh],
    faces: triangularFaces(model.faces),
    boneWxyzs,
    bonePositions,
    skinWeights: viserSkinWeights(model.skinWeights),
    color: options.color ?? [180, 180, 180],
  });
  return new ViserBodyModelHandle(model, pose, root, mesh);
}

export function createBodyModel<TParams extends BodyModelParams>(spec: {
  modelName: string;
  faces: BodyModelLike<TParams>["faces"];
  skinWeights: BodyModelLike<TParams>["skinWeights"];
  poseParameterNames: readonly (keyof TParams & string)[];
  getRestPose: () => TParams;
  forward: (params: TParams) => BodyModelForwardOutput;
  getBindParams?: (params: TParams) => TParams;
  isRigidBody?: boolean;
}): BodyModelLike<TParams> {
  return {
    modelName: spec.modelName,
    isRigidBody: spec.isRigidBody ?? false,
    faces: spec.faces,
    skinWeights: spec.skinWeights,
    poseParameterNames: spec.poseParameterNames,
    getRestPose: spec.getRestPose,
    getBindParams: spec.getBindParams ?? ((params) => bindParamsFromRest(spec.getRestPose(), params, spec.poseParameterNames)),
    forwardSkeleton: (params) => [...spec.forward(params).skeleton],
    forwardVertices: (params) => [...spec.forward(params).mesh],
    forward: spec.forward,
  };
}

export const bodyModelsViserPlugin = {
  addBodyModel,
  createBodyModel,
  ViserBodyModelHandle,
};

function bindParamsFromRest<TParams extends BodyModelParams>(
  rest: TParams,
  params: TParams,
  poseParameterNames: readonly string[],
): TParams {
  const bind = cloneParams(rest);
  for (const [name, value] of Object.entries(params) as [keyof TParams & string, NumericArray][]) {
    if (name in bind && !poseParameterNames.includes(name)) {
      bind[name] = cloneNumericArray(value) as TParams[keyof TParams & string];
    }
  }
  return bind;
}

function bonePoses(skeleton: readonly Mat4[]): { boneWxyzs: QuatWxyz[]; bonePositions: Vec3[] } {
  return {
    boneWxyzs: skeleton.map((transform) => rotmatToWxyz(transform)),
    bonePositions: skeleton.map((transform) => [transform[0][3], transform[1][3], transform[2][3]]),
  };
}

function triangularFaces(faces: BodyModelLike["faces"]): [number, number, number][] {
  const triangles: [number, number, number][] = [];
  for (const face of faces) {
    if (face.length === 3) {
      triangles.push([face[0], face[1], face[2]]);
    } else if (face.length === 4) {
      triangles.push([face[0], face[1], face[2]], [face[0], face[2], face[3]]);
    } else {
      throw new Error("Expected triangular or quad faces.");
    }
  }
  return triangles;
}

function viserSkinWeights(skinWeights: BodyModelLike["skinWeights"]): number[][] {
  return skinWeights.map((row) => {
    const weights = Array.from(row);
    const keep = topSkinWeightIndices(weights);
    let sum = 0.0;
    for (let i = 0; i < weights.length; i++) {
      if (!keep.has(i)) {
        weights[i] = 0.0;
      }
      sum += weights[i]!;
    }
    return sum > 0.0 ? weights.map((weight) => weight / sum) : weights;
  });
}

function topSkinWeightIndices(weights: number[]): Set<number> {
  const ranked = weights.map((weight, index) => ({ weight, index }));
  ranked.sort((a, b) => b.weight - a.weight);
  return new Set(ranked.slice(0, 4).map((item) => item.index));
}

function rotmatToWxyz(transform: Mat4): QuatWxyz {
  const m00 = transform[0][0];
  const m01 = transform[0][1];
  const m02 = transform[0][2];
  const m10 = transform[1][0];
  const m11 = transform[1][1];
  const m12 = transform[1][2];
  const m20 = transform[2][0];
  const m21 = transform[2][1];
  const m22 = transform[2][2];
  const trace = m00 + m11 + m22;
  let w: number;
  let x: number;
  let y: number;
  let z: number;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1.0) * 2.0;
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1.0 + m00 - m11 - m22) * 2.0;
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1.0 + m11 - m00 - m22) * 2.0;
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1.0 + m22 - m00 - m11) * 2.0;
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }
  const norm = Math.hypot(w, x, y, z);
  return [w / norm, x / norm, y / norm, z / norm];
}

function cloneParams<TParams extends BodyModelParams>(params: TParams): TParams {
  const out: BodyModelParams = {};
  for (const [key, value] of Object.entries(params)) {
    out[key] = cloneNumericArray(value);
  }
  return out as TParams;
}

function cloneNumericArray(value: NumericArray): NumericArray {
  if (ArrayBuffer.isView(value)) {
    return value.slice() as Float32Array | Float64Array;
  }
  return value.map((item) => (Array.isArray(item) ? item.slice() : item)) as number[] | number[][];
}

function numericArrayEqual(a: NumericArray, b: NumericArray): boolean {
  const flatA = flattenNumericArray(a);
  const flatB = flattenNumericArray(b);
  if (flatA.length !== flatB.length) {
    return false;
  }
  for (let i = 0; i < flatA.length; i++) {
    if (flatA[i] !== flatB[i]) {
      return false;
    }
  }
  return true;
}

function flattenNumericArray(value: NumericArray): readonly number[] {
  if (ArrayBuffer.isView(value)) {
    return Array.from(value);
  }
  return value.flatMap((item) => (Array.isArray(item) ? item : [item]));
}
