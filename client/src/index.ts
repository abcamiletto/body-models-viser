export let version = "";
export let buildId = "";

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

type RemoveSceneNodeMessage = { type: "RemoveSceneNodeMessage"; name: string };
type Message = AddSmplMessage | PoseMessage | MeshMessage | RemoveSceneNodeMessage | { type: string };

type ViewerLike = {
  mutable: {
    current: {
      messageQueue: Message[];
      sendMessage(message: { type: "BodyModelsViserReadyMessage" }): void;
    };
  };
};

type WasmExports = {
  memory: WebAssembly.Memory;
  alloc(size: number): number;
  wasm_free(ptr: number, len: number): void;
  smpl_forward_vertices(
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
    outputVerticesPtr: number,
  ): void;
};

type WasmBuffer = { ptr: number; len: number; byteLen: number };
type MeshState = {
  vertexCount: number;
  lbsWeights: WasmBuffer;
  restJoints: WasmBuffer;
  restVertices: WasmBuffer;
  jointTransforms: WasmBuffer;
  poseOffsets: WasmBuffer;
  globalRotation: WasmBuffer;
  globalTranslation: WasmBuffer;
  outputVertices: WasmBuffer;
  faces: Uint32Array;
  props: MeshProps;
};

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
    this.ready();
  }

  ready(): void {
    this.getViewer().mutable.current.sendMessage({ type: "BodyModelsViserReadyMessage" });
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
    if (message.type === "RemoveSceneNodeMessage") {
      this.remove(message as RemoveSceneNodeMessage);
    }
    return false;
  }

  private addSmpl(message: AddSmplMessage): void {
    const existing = this.meshes.get(message.name);
    if (existing !== undefined) {
      this.freeMesh(existing);
    }
    this.meshes.set(message.name, {
      vertexCount: message.vertex_count,
      lbsWeights: this.copyToWasm(message.lbs_weights),
      restJoints: this.copyToWasm(message.rest_joints),
      restVertices: this.copyToWasm(message.rest_vertices),
      jointTransforms: this.copyToWasm(message.joint_transforms),
      poseOffsets: this.copyToWasm(message.pose_offsets),
      globalRotation: this.copyToWasm(message.global_rotation),
      globalTranslation: this.copyToWasm(message.global_translation),
      outputVertices: this.allocF32(message.vertex_count * 3),
      faces: message.faces,
      props: message.props,
    });
    this.pushMesh(message.name);
  }

  private remove(message: RemoveSceneNodeMessage): void {
    const mesh = this.meshes.get(message.name);
    if (mesh !== undefined) {
      this.freeMesh(mesh);
      this.meshes.delete(message.name);
    }
  }

  private setPose(message: PoseMessage): void {
    const mesh = this.meshes.get(message.name);
    if (mesh === undefined) {
      throw new Error(`SMPL ${message.name} has not been created.`);
    }
    if (message.rest_joints !== null && message.rest_vertices !== null) {
      this.copyIntoWasm(mesh.restJoints, message.rest_joints);
      this.copyIntoWasm(mesh.restVertices, message.rest_vertices);
    }
    if (message.joint_transforms !== null && message.pose_offsets !== null) {
      this.copyIntoWasm(mesh.jointTransforms, message.joint_transforms);
      this.copyIntoWasm(mesh.poseOffsets, message.pose_offsets);
    }
    this.copyIntoWasm(mesh.globalRotation, message.global_rotation);
    this.copyIntoWasm(mesh.globalTranslation, message.global_translation);
    this.pushMesh(message.name);
  }

  private pushMesh(name: string): void {
    const mesh = this.meshes.get(name);
    if (mesh === undefined) {
      throw new Error(`SMPL ${name} has not been created.`);
    }
    const wasm = this.requireWasm();
    wasm.smpl_forward_vertices(
      mesh.lbsWeights.ptr,
      mesh.lbsWeights.len,
      mesh.restJoints.ptr,
      mesh.restJoints.len,
      mesh.restVertices.ptr,
      mesh.restVertices.len,
      mesh.jointTransforms.ptr,
      mesh.jointTransforms.len,
      mesh.poseOffsets.ptr,
      mesh.poseOffsets.len,
      mesh.globalRotation.ptr,
      mesh.globalTranslation.ptr,
      mesh.outputVertices.ptr,
    );
    const vertices = this.copyOutput(mesh.outputVertices);
    this.getViewer().mutable.current.messageQueue.push({
      type: "MeshMessage",
      name,
      props: { ...mesh.props, vertices, faces: mesh.faces },
    });
  }

  private copyToWasm(array: Float32Array): WasmBuffer {
    const buffer = this.allocF32(array.length);
    this.copyIntoWasm(buffer, array);
    return buffer;
  }

  private allocF32(len: number): WasmBuffer {
    const wasm = this.requireWasm();
    return { ptr: wasm.alloc(len * 4), len, byteLen: len * 4 };
  }

  private copyIntoWasm(buffer: WasmBuffer, array: Float32Array): void {
    if (array.length !== buffer.len) {
      throw new Error(`Expected ${buffer.len} f32 values, received ${array.length}.`);
    }
    const wasm = this.requireWasm();
    new Float32Array(wasm.memory.buffer, buffer.ptr, buffer.len).set(array);
  }

  private copyOutput(buffer: WasmBuffer): Float32Array {
    const wasm = this.requireWasm();
    return new Float32Array(new Float32Array(wasm.memory.buffer, buffer.ptr, buffer.len));
  }

  private freeMesh(mesh: MeshState): void {
    this.freeBuffer(mesh.lbsWeights);
    this.freeBuffer(mesh.restJoints);
    this.freeBuffer(mesh.restVertices);
    this.freeBuffer(mesh.jointTransforms);
    this.freeBuffer(mesh.poseOffsets);
    this.freeBuffer(mesh.globalRotation);
    this.freeBuffer(mesh.globalTranslation);
    this.freeBuffer(mesh.outputVertices);
  }

  private freeBuffer(buffer: WasmBuffer): void {
    this.requireWasm().wasm_free(buffer.ptr, buffer.byteLen);
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

export function install(wasmBase64: string, runtimeVersion: string, runtimeBuildId: string): void {
  version = runtimeVersion;
  buildId = runtimeBuildId;
  runtime.install(wasmBase64);
}

export function ready(): void {
  runtime.ready();
}
