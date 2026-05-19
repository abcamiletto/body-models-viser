// AUTOMATICALLY GENERATED body-models viser interfaces, from Python definitions.
// This file should not be manually modified.

export type NumericArray = number[] | number[][] | Float32Array | Float64Array;
export type BodyModelParams = Record<string, NumericArray>;
export type Vec3 = [number, number, number];
export type QuatWxyz = [number, number, number, number];
export type Face = [number, number, number] | [number, number, number, number];
export type Mat4 = [[number, number, number, number], [number, number, number, number], [number, number, number, number], [number, number, number, number]];

export interface BodyModelForwardOutput {
  skeleton: readonly Mat4[];
  mesh: readonly Vec3[];
}

export interface BodyModelSceneFrameOptions {
  showAxes: boolean;
}

export interface BodyModelSceneSkinnedMeshOptions {
  vertices: readonly Vec3[];
  faces: readonly [number, number, number][];
  boneWxyzs: readonly QuatWxyz[];
  bonePositions: readonly Vec3[];
  skinWeights: readonly (readonly number[])[];
  color: Vec3;
}

export interface ViserFrameHandle {
  name: string;
  wxyz: QuatWxyz;
  position: Vec3;
  visible: boolean;
  remove(): void;
}

export interface ViserBoneHandle {
  wxyz: QuatWxyz;
  position: Vec3;
}

export interface ViserSkinnedMeshHandle {
  vertices: readonly Vec3[];
  bones: readonly ViserBoneHandle[];
  remove(): void;
}

export interface BodyModelScene {
  addFrame(name: string, options: BodyModelSceneFrameOptions): ViserFrameHandle;
  addMeshSkinned(name: string, options: BodyModelSceneSkinnedMeshOptions): ViserSkinnedMeshHandle;
}

export interface BodyModelLike<TParams extends BodyModelParams = BodyModelParams> {
  modelName: string;
  isRigidBody?: boolean;
  poseParameterNames: readonly (keyof TParams & string)[];
  faces: readonly Face[];
  skinWeights: readonly (readonly number[])[];
  getRestPose(): TParams;
  getBindParams(params: TParams): TParams;
  forwardSkeleton(params: TParams): Mat4[];
  forwardVertices(params: TParams): Vec3[];
  forward(params: TParams): BodyModelForwardOutput;
}
