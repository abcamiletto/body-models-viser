import ndarray from "ndarray";

type MeshProps = {
  color: [number, number, number];
  wireframe: boolean;
  opacity: number | null;
  flat_shading: boolean;
  side: "front" | "back" | "double";
  material: "standard" | "toon3" | "toon5";
  scale: number | [number, number, number];
  cast_shadow: boolean;
  receive_shadow: boolean | number;
};

type AddSmplMessage = {
  type: "BodyModelsViserSmplMessage";
  name: string;
  vertex_count: number;
  face_count: number;
  lbs_weights: Float32Array;
  faces: Uint32Array;
  rest_joints: Float32Array;
  rest_vertices: Float32Array;
  joint_transforms: Float32Array;
  pose_offsets: Float32Array;
  global_rotation: Float32Array;
  global_translation: Float32Array;
  props: MeshProps;
};

type PoseMessage = {
  type: "BodyModelsViserPoseMessage";
  name: string;
  rest_joints: Float32Array | null;
  rest_vertices: Float32Array | null;
  joint_transforms: Float32Array | null;
  pose_offsets: Float32Array | null;
  global_rotation: Float32Array;
  global_translation: Float32Array;
};

type MeshMessage = {
  type: "MeshMessage";
  name: string;
  props: MeshProps & {
    vertices: Float32Array;
    faces: Uint32Array;
  };
};

type Message = AddSmplMessage | PoseMessage | MeshMessage | { type: string };

type ViewerLike = {
  mutable: {
    current: {
      messageQueue: Message[];
    };
  };
};

type WasmExports = {
  memory: WebAssembly.Memory;
  alloc(size: number): number;
  wasm_free(ptr: number, len: number): void;
  output_free(handle: bigint): void;
  smpl_create(
    lbsWeightsPtr: number,
    lbsWeightsLen: number,
    restJointsPtr: number,
    restJointsLen: number,
    restVerticesPtr: number,
    restVerticesLen: number,
    jointTransformsPtr: number,
    jointTransformsLen: number,
    poseOffsetsPtr: number,
    poseOffsetsLen: number,
    globalRotationPtr: number,
    globalTranslationPtr: number,
  ): number;
  smpl_set_identity(
    model: number,
    restJointsPtr: number,
    restJointsLen: number,
    restVerticesPtr: number,
    restVerticesLen: number,
  ): void;
  smpl_set_pose(
    model: number,
    jointTransformsPtr: number,
    jointTransformsLen: number,
    poseOffsetsPtr: number,
    poseOffsetsLen: number,
  ): void;
  smpl_set_global(model: number, globalRotationPtr: number, globalTranslationPtr: number): void;
  smpl_forward(model: number): bigint;
};

type WasmInput = { ptr: number; byteLen: number; len: number };
type MeshState = { model: number; vertexCount: number; faces: Uint32Array; props: MeshProps };

class BodyModelsViserRuntime {
  private wasm: WasmExports | null = null;
  private viewer: ViewerLike | null = null;
  private meshes = new Map<string, MeshState>();

  install(wasmBase64: string): void {
    const bytes = Uint8Array.from(atob(wasmBase64), (char) => char.charCodeAt(0));
    const module = new WebAssembly.Module(bytes.buffer as ArrayBuffer);
    const instance = new WebAssembly.Instance(module);
    this.wasm = instance.exports as WasmExports;
    this.patchMessageQueue();
  }

  consume(message: Message): boolean {
    if (message.type === "BodyModelsViserSmplMessage") {
      this.addSmpl(message as AddSmplMessage);
      return true;
    }
    if (message.type === "BodyModelsViserPoseMessage") {
      this.setPose(message as PoseMessage);
      return true;
    }
    return false;
  }

  private addSmpl(message: AddSmplMessage): void {
    const wasm = this.requireWasm();
    const lbsWeights = this.writeArray(float32(message.lbs_weights, [message.vertex_count, 24]));
    const restJoints = this.writeArray(float32(message.rest_joints, [24, 3]));
    const restVertices = this.writeArray(float32(message.rest_vertices, [message.vertex_count, 3]));
    const jointTransforms = this.writeArray(float32(message.joint_transforms, [24, 4, 4]));
    const poseOffsets = this.writeArray(float32(message.pose_offsets, [message.vertex_count, 3]));
    const globalRotation = this.writeArray(float32(message.global_rotation, [3]));
    const globalTranslation = this.writeArray(float32(message.global_translation, [3]));
    const model = wasm.smpl_create(
      lbsWeights.ptr,
      lbsWeights.len,
      restJoints.ptr,
      restJoints.len,
      restVertices.ptr,
      restVertices.len,
      jointTransforms.ptr,
      jointTransforms.len,
      poseOffsets.ptr,
      poseOffsets.len,
      globalRotation.ptr,
      globalTranslation.ptr,
    );
    [lbsWeights, restJoints, restVertices, jointTransforms, poseOffsets, globalRotation, globalTranslation].forEach(
      (input) => this.free(input),
    );
    this.meshes.set(message.name, {
      model,
      vertexCount: message.vertex_count,
      faces: uint32(message.faces, [message.face_count, 3]).data,
      props: message.props,
    });
    this.pushMesh(message.name);
  }

  private setPose(message: PoseMessage): void {
    const mesh = this.meshes.get(message.name);
    if (mesh === undefined) {
      throw new Error(`SMPL ${message.name} has not been created.`);
    }
    const wasm = this.requireWasm();
    if (message.rest_joints !== null && message.rest_vertices !== null) {
      const restJoints = this.writeArray(float32(message.rest_joints, [24, 3]));
      const restVertices = this.writeArray(float32(message.rest_vertices, [mesh.vertexCount, 3]));
      wasm.smpl_set_identity(mesh.model, restJoints.ptr, restJoints.len, restVertices.ptr, restVertices.len);
      this.free(restJoints);
      this.free(restVertices);
    }
    if (message.joint_transforms !== null && message.pose_offsets !== null) {
      const jointTransforms = this.writeArray(float32(message.joint_transforms, [24, 4, 4]));
      const poseOffsets = this.writeArray(float32(message.pose_offsets, [message.pose_offsets.length / 3, 3]));
      wasm.smpl_set_pose(mesh.model, jointTransforms.ptr, jointTransforms.len, poseOffsets.ptr, poseOffsets.len);
      this.free(jointTransforms);
      this.free(poseOffsets);
    }
    const globalRotation = this.writeArray(float32(message.global_rotation, [3]));
    const globalTranslation = this.writeArray(float32(message.global_translation, [3]));
    wasm.smpl_set_global(mesh.model, globalRotation.ptr, globalTranslation.ptr);
    this.free(globalRotation);
    this.free(globalTranslation);
    this.pushMesh(message.name);
  }

  private pushMesh(name: string): void {
    const mesh = this.meshes.get(name);
    if (mesh === undefined) {
      throw new Error(`SMPL ${name} has not been created.`);
    }
    const wasm = this.requireWasm();
    const output = wasm.smpl_forward(mesh.model);
    const vertices = this.readFloat32Output(output);
    wasm.output_free(output);
    this.getViewer().mutable.current.messageQueue.push({
      type: "MeshMessage",
      name,
      props: { ...mesh.props, vertices, faces: mesh.faces },
    });
  }

  private writeArray(array: ndarray.NdArray<Float32Array | Uint32Array>): WasmInput {
    const wasm = this.requireWasm();
    const bytes = new Uint8Array(array.data.buffer, array.data.byteOffset, array.data.byteLength);
    const ptr = wasm.alloc(bytes.byteLength);
    new Uint8Array(wasm.memory.buffer, ptr, bytes.byteLength).set(bytes);
    return { ptr, byteLen: bytes.byteLength, len: array.data.length };
  }

  private readFloat32Output(handle: bigint): Float32Array {
    const wasm = this.requireWasm();
    const ptr = Number(handle >> 32n);
    const len = Number(handle & 0xffffffffn);
    return new Float32Array(new Float32Array(wasm.memory.buffer, ptr, len));
  }

  private free(input: WasmInput): void {
    this.requireWasm().wasm_free(input.ptr, input.byteLen);
  }

  private requireWasm(): WasmExports {
    if (this.wasm === null) {
      throw new Error("body-models-viser WASM is not installed.");
    }
    return this.wasm;
  }

  private getViewer(): ViewerLike {
    this.viewer ??= findViewer();
    return this.viewer;
  }

  private patchMessageQueue(): void {
    const viewer = this.getViewer();
    const queue = viewer.mutable.current.messageQueue;
    const push = queue.push.bind(queue);
    queue.push = (...messages: Message[]) => push(...messages.filter((message) => !this.consume(message)));
  }
}

function float32(data: Float32Array, shape: number[]): ndarray.NdArray<Float32Array> {
  return ndarray(data, shape);
}

function uint32(data: Uint32Array, shape: number[]): ndarray.NdArray<Uint32Array> {
  return ndarray(data, shape);
}

type ReactFiberNode = {
  memoizedProps?: {
    value?: Partial<ViewerLike> & Record<string, unknown>;
  };
  child?: ReactFiberNode | null;
  sibling?: ReactFiberNode | null;
};

function findViewer(): ViewerLike {
  const root = document.getElementById("root") as Record<string, unknown> | null;
  const key = root ? Object.keys(root).find((candidate) => candidate.startsWith("__reactContainer$")) : undefined;
  const stack = key ? [root![key] as ReactFiberNode] : [];
  const seen = new Set<unknown>();

  while (stack.length > 0) {
    const fiber = stack.pop();
    if (!fiber || seen.has(fiber)) {
      continue;
    }
    seen.add(fiber);
    if (isViewerLike(fiber.memoizedProps?.value)) {
      return fiber.memoizedProps.value;
    }
    if (fiber.child) {
      stack.push(fiber.child);
    }
    if (fiber.sibling) {
      stack.push(fiber.sibling);
    }
  }
  throw new Error("Could not locate the viser viewer.");
}

function isViewerLike(value: unknown): value is ViewerLike {
  return (
    typeof value === "object" &&
    value !== null &&
    Array.isArray((value as Partial<ViewerLike>).mutable?.current?.messageQueue)
  );
}

const runtime = new BodyModelsViserRuntime();

export function install(wasmBase64: string): void {
  runtime.install(wasmBase64);
}
