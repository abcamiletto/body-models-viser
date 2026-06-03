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

type ModelMessage = {
  type: "BodyModelsViserModelMessage";
  name: string;
  vertex_count: number;
  lbs_weights: Float32Array;
  faces: Uint32Array;
  rest_vertices: Float32Array;
  skinning_transforms: Float32Array;
  pose_offsets: Float32Array;
  global_rotation: Float32Array;
  global_translation: Float32Array;
  props: MeshProps;
};

type PoseMessage = {
  type: "BodyModelsViserPoseMessage";
  name: string;
  rest_vertices: Float32Array | null;
  skinning_transforms: Float32Array | null;
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
type Message = ModelMessage | PoseMessage | MeshMessage | RemoveSceneNodeMessage | { type: string };

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
  forward_vertices(
    lbsWeightsPtr: number,
    lbsWeightsLen: number,
    restVerticesPtr: number,
    restVerticesLen: number,
    skinningTransformsPtr: number,
    skinningTransformsLen: number,
    poseOffsetsPtr: number,
    poseOffsetsLen: number,
    globalRotationPtr: number,
    globalTranslationPtr: number,
    outputVerticesPtr: number,
  ): void;
};

type WasmBuffer = { ptr: number; len: number; byteLen: number };
type MeshState = {
  lbsWeights: WasmBuffer;
  restVertices: WasmBuffer;
  skinningTransforms: WasmBuffer;
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
    if (message.type === "BodyModelsViserModelMessage") {
      this.addModel(message as ModelMessage);
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

  private addModel(message: ModelMessage): void {
    const existing = this.meshes.get(message.name);
    if (existing !== undefined) {
      this.freeMesh(existing);
    }
    const mesh = {
      lbsWeights: this.copyToWasm(message.lbs_weights),
      restVertices: this.copyToWasm(message.rest_vertices),
      skinningTransforms: this.copyToWasm(message.skinning_transforms),
      poseOffsets: this.copyToWasm(message.pose_offsets),
      globalRotation: this.copyToWasm(message.global_rotation),
      globalTranslation: this.copyToWasm(message.global_translation),
      outputVertices: this.allocF32(message.vertex_count * 3),
      faces: message.faces,
      props: message.props,
    };
    this.meshes.set(message.name, mesh);
    this.pushMesh(message.name, mesh);
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
      throw new Error(`Body model ${message.name} has not been created.`);
    }
    if (message.rest_vertices !== null) {
      this.copyIntoWasm(mesh.restVertices, message.rest_vertices);
    }
    if (message.skinning_transforms !== null) {
      this.copyIntoWasm(mesh.skinningTransforms, message.skinning_transforms);
      this.copyIntoWasm(mesh.poseOffsets, message.pose_offsets!);
    }
    this.copyIntoWasm(mesh.globalRotation, message.global_rotation);
    this.copyIntoWasm(mesh.globalTranslation, message.global_translation);
    this.pushMesh(message.name, mesh);
  }

  private pushMesh(name: string, mesh: MeshState): void {
    const wasm = this.requireWasm();
    wasm.forward_vertices(
      mesh.lbsWeights.ptr,
      mesh.lbsWeights.len,
      mesh.restVertices.ptr,
      mesh.restVertices.len,
      mesh.skinningTransforms.ptr,
      mesh.skinningTransforms.len,
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
    this.freeBuffer(mesh.restVertices);
    this.freeBuffer(mesh.skinningTransforms);
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
    queue.push = (...messages: Message[]) => {
      const forwarded = messages.filter((message) => !this.consume(message));
      return push(...forwarded);
    };
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
  const containerKey = root
    ? Object.keys(root).find((candidate) => candidate.startsWith("__reactContainer$"))
    : undefined;
  const stack = containerKey ? [root![containerKey] as ReactFiberNode] : [];
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

export function ready(): void {
  runtime.ready();
}
