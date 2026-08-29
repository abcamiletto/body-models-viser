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

type CorrectiveMessage =
  | {
      corrective_format: null;
      corrective_values: null;
      corrective_scales: null;
      corrective_offsets: null;
      corrective_indices: null;
    }
  | {
      corrective_format: "dense";
      corrective_values: Int16Array;
      corrective_scales: Float32Array;
      corrective_offsets: null;
      corrective_indices: null;
    }
  | {
      corrective_format: "sparse";
      corrective_values: Int16Array;
      corrective_scales: Float32Array;
      corrective_offsets: Uint32Array;
      corrective_indices: Uint16Array;
    };

type AssetMessage = CorrectiveMessage & {
  type: "BodyModelsViserAssetMessage";
  asset_id: number;
  faces: Uint32Array;
  skin_weight_offsets: Uint32Array;
  skin_weight_indices: Uint16Array;
  skin_weight_values: Float32Array;
};

type ModelMessage = {
  type: "BodyModelsViserModelMessage";
  name: string;
  asset_id: number;
  rest_vertices: Float32Array;
  skinning_transforms: Float32Array;
  pose_coefficients: Float32Array | null;
  global_rotation: Float32Array;
  global_translation: Float32Array;
  wxyz: [number, number, number, number];
  position: [number, number, number];
  visible: boolean;
  props: MeshProps;
};

type IdentityMessage = {
  type: "BodyModelsViserIdentityMessage";
  name: string;
  rest_vertices: Float32Array;
  skinning_transforms: Float32Array;
  pose_coefficients: Float32Array | null;
};

type PoseMessage = {
  type: "BodyModelsViserPoseMessage";
  name: string;
  skinning_transforms: Float32Array;
  pose_coefficients: Float32Array | null;
};

type TransformMessage = {
  type: "BodyModelsViserTransformMessage";
  name: string;
  global_rotation: Float32Array;
  global_translation: Float32Array;
};

type MeshMessage = {
  type: "MeshMessage";
  name: string;
  props: MeshProps & { vertices: Float32Array; faces: Uint32Array };
};

type RemoveSceneNodeMessage = { type: "RemoveSceneNodeMessage"; name: string };
type SetSceneNodeVisibilityMessage = {
  type: "SetSceneNodeVisibilityMessage";
  name: string;
  visible: boolean;
};
type SetOrientationMessage = {
  type: "SetOrientationMessage";
  name: string;
  wxyz: [number, number, number, number];
};
type SetPositionMessage = {
  type: "SetPositionMessage";
  name: string;
  position: [number, number, number];
};
type Message =
  | AssetMessage
  | ModelMessage
  | IdentityMessage
  | PoseMessage
  | TransformMessage
  | MeshMessage
  | RemoveSceneNodeMessage
  | SetSceneNodeVisibilityMessage
  | SetOrientationMessage
  | SetPositionMessage
  | { type: string };

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
  compute_dense_pose_offsets(
    valuesPtr: number,
    valuesLen: number,
    scalesPtr: number,
    scalesLen: number,
    coefficientsPtr: number,
    coefficientsLen: number,
    outputPtr: number,
    outputLen: number,
  ): void;
  compute_sparse_pose_offsets(
    offsetsPtr: number,
    offsetsLen: number,
    indicesPtr: number,
    indicesLen: number,
    valuesPtr: number,
    valuesLen: number,
    scalesPtr: number,
    scalesLen: number,
    coefficientsPtr: number,
    coefficientsLen: number,
    outputPtr: number,
    outputLen: number,
  ): void;
  forward_vertices_sparse(
    weightOffsetsPtr: number,
    weightOffsetsLen: number,
    weightIndicesPtr: number,
    weightIndicesLen: number,
    weightValuesPtr: number,
    weightValuesLen: number,
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
type PreloadState = { pending: Message[]; restore(): void };
type GpuBuffer = any;
type GpuDevice = any;

type GpuContext = { device: GpuDevice; pipeline: any };
type GpuBatch = {
  meshes: MeshState[];
  restVersions: number[];
  dynamic: GpuBuffer;
  output: GpuBuffer;
  readback: GpuBuffer;
  params: GpuBuffer;
  bindGroup: any;
  coefficientOffset: number;
  restOffset: number;
  transformOffset: number;
  globalOffset: number;
};
type GpuAsset = {
  context: GpuContext;
  correctiveValues: GpuBuffer;
  correctiveScales: GpuBuffer;
  correctiveOffsets: GpuBuffer;
  correctiveIndices: GpuBuffer;
  weightOffsets: GpuBuffer;
  weightIndices: GpuBuffer;
  weightValues: GpuBuffer;
  batch: GpuBatch | null;
};

type DenseCorrectives = {
  format: "dense";
  values: WasmBuffer | null;
  scales: WasmBuffer | null;
  valueValues: Int16Array;
  scaleValues: Float32Array;
};

type SparseCorrectives = {
  format: "sparse";
  values: WasmBuffer | null;
  scales: WasmBuffer | null;
  offsets: WasmBuffer | null;
  indices: WasmBuffer | null;
  valueValues: Int16Array;
  scaleValues: Float32Array;
  offsetValues: Uint32Array;
  indexValues: Uint16Array;
};

type Correctives = DenseCorrectives | SparseCorrectives;

type AssetState = {
  id: number;
  faces: Uint32Array;
  weightOffsets: WasmBuffer;
  weightIndices: WasmBuffer;
  weightValues: WasmBuffer;
  correctives: Correctives | null;
  weightOffsetValues: Uint32Array;
  weightIndexValues: Uint16Array;
  weightValueValues: Float32Array;
  meshes: Set<MeshState>;
  dirty: Set<MeshState>;
  scheduled: boolean;
  busy: boolean;
  disposeWhenIdle: boolean;
  gpu: Promise<GpuAsset | null> | null;
};

type MeshState = {
  name: string;
  asset: AssetState;
  restVertices: WasmBuffer;
  skinningTransforms: WasmBuffer;
  poseCoefficients: WasmBuffer | null;
  poseOffsets: WasmBuffer;
  globalRotation: WasmBuffer;
  globalTranslation: WasmBuffer;
  outputVertices: WasmBuffer;
  restValues: Float32Array;
  transformValues: Float32Array;
  coefficientValues: Float32Array | null;
  globalRotationValues: Float32Array;
  globalTranslationValues: Float32Array;
  props: MeshProps;
  restVersion: number;
  sceneStateSent: boolean;
  wxyz: [number, number, number, number];
  position: [number, number, number];
  visible: boolean;
};

const CORRECTIVE_SHADER = /* wgsl */ `
struct Params {
  vertex_count: u32,
  coefficient_count: u32,
  joint_count: u32,
  body_count: u32,
  coefficient_offset: u32,
  rest_offset: u32,
  transform_offset: u32,
  global_offset: u32,
  corrective_format: u32,
  _padding_0: u32,
  _padding_1: u32,
  _padding_2: u32,
}

@group(0) @binding(0) var<storage, read> corrective_values: array<u32>;
@group(0) @binding(1) var<storage, read> corrective_scales: array<f32>;
@group(0) @binding(2) var<storage, read> corrective_offsets: array<u32>;
@group(0) @binding(3) var<storage, read> corrective_indices: array<u32>;
@group(0) @binding(4) var<storage, read> weight_offsets: array<u32>;
@group(0) @binding(5) var<storage, read> weight_indices: array<u32>;
@group(0) @binding(6) var<storage, read> weight_values: array<f32>;
@group(0) @binding(7) var<storage, read> dynamic: array<f32>;
@group(0) @binding(8) var<storage, read_write> output: array<f32>;
@group(0) @binding(9) var<uniform> params: Params;

fn corrective_value(index: u32) -> f32 {
  let word = corrective_values[index >> 1u];
  let raw = (word >> ((index & 1u) * 16u)) & 65535u;
  var signed = i32(raw);
  if (raw >= 32768u) { signed -= 65536; }
  return f32(signed);
}

fn corrective_offset(coordinate: u32, coefficient_base: u32) -> f32 {
  var correction = 0.0;
  if (params.corrective_format == 0u) {
    let value_base = coordinate * params.coefficient_count;
    for (var coefficient = 0u;
         coefficient < params.coefficient_count;
         coefficient++) {
      correction += dynamic[coefficient_base + coefficient]
        * corrective_value(value_base + coefficient);
    }
  } else {
    for (var entry = corrective_offsets[coordinate];
         entry < corrective_offsets[coordinate + 1u];
         entry++) {
      let coefficient = corrective_indices[entry];
      correction += dynamic[coefficient_base + coefficient]
        * corrective_value(entry);
    }
  }
  return correction * corrective_scales[coordinate];
}

fn rotate_axis_angle(point: vec3<f32>, rotation: vec3<f32>) -> vec3<f32> {
  let angle = length(rotation);
  if (angle < 1e-8) { return point; }
  let axis = rotation / angle;
  let c = cos(angle);
  let s = sin(angle);
  return point * c + cross(axis, point) * s + axis * dot(axis, point) * (1.0 - c);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let vertex = id.x;
  let body = id.y;
  if (vertex >= params.vertex_count || body >= params.body_count) { return; }

  let coordinate_count = params.vertex_count * 3u;
  let coordinate = vertex * 3u;
  let coefficient_base = params.coefficient_offset + body * params.coefficient_count;
  let rest_base = params.rest_offset + body * coordinate_count + coordinate;
  var point = vec3<f32>(
    dynamic[rest_base], dynamic[rest_base + 1u], dynamic[rest_base + 2u]
  );
  for (var component = 0u; component < 3u; component++) {
    point[component] += corrective_offset(coordinate + component, coefficient_base);
  }

  let transform_body_base = params.transform_offset
    + body * params.joint_count * 16u;
  var skinned = vec3<f32>(0.0);
  for (var influence = weight_offsets[vertex];
       influence < weight_offsets[vertex + 1u];
       influence++) {
    let joint = weight_indices[influence];
    let matrix = transform_body_base + joint * 16u;
    let transformed = vec3<f32>(
      dynamic[matrix] * point.x + dynamic[matrix + 1u] * point.y
        + dynamic[matrix + 2u] * point.z + dynamic[matrix + 3u],
      dynamic[matrix + 4u] * point.x + dynamic[matrix + 5u] * point.y
        + dynamic[matrix + 6u] * point.z + dynamic[matrix + 7u],
      dynamic[matrix + 8u] * point.x + dynamic[matrix + 9u] * point.y
        + dynamic[matrix + 10u] * point.z + dynamic[matrix + 11u]
    );
    skinned += transformed * weight_values[influence];
  }

  let global = params.global_offset + body * 6u;
  let rotation = vec3<f32>(dynamic[global], dynamic[global + 1u], dynamic[global + 2u]);
  let translation = vec3<f32>(
    dynamic[global + 3u], dynamic[global + 4u], dynamic[global + 5u]
  );
  let posed = rotate_axis_angle(skinned, rotation) + translation;
  let output_base = body * coordinate_count + coordinate;
  output[output_base] = posed.x;
  output[output_base + 1u] = posed.y;
  output[output_base + 2u] = posed.z;
}
`;

class BodyModelsViserRuntime {
  private wasm: WasmExports | null = null;
  private viewer: ViewerLike | null = null;
  private assets = new Map<number, AssetState>();
  private meshes = new Map<string, MeshState>();
  private gpuContext: Promise<GpuContext | null> | null = null;
  private animationFrames: number[] = [];
  private modelRenders: number[] = [];
  private correctiveBatchTimes: number[] = [];
  private correctiveBackend: "none" | "webgpu" | "wasm" = "none";

  install(wasmBase64: string): void {
    const bytes = Uint8Array.from(atob(wasmBase64), (char) => char.charCodeAt(0));
    const module = new WebAssembly.Module(bytes.buffer as ArrayBuffer);
    const instance = new WebAssembly.Instance(module);
    this.wasm = instance.exports as WasmExports;
    this.patchMessageQueue();
    this.drainPreload();
    this.trackAnimationFrames();
    this.ready();
  }

  ready(): void {
    this.getViewer().mutable.current.sendMessage({ type: "BodyModelsViserReadyMessage" });
  }

  stats(): Record<string, number | string> {
    const now = performance.now();
    this.pruneTimings(now);
    const meanBatchTime = this.correctiveBatchTimes.length
      ? this.correctiveBatchTimes.reduce((sum, value) => sum + value, 0) /
        this.correctiveBatchTimes.length
      : 0;
    return {
      bodies: this.meshes.size,
      renderFps: this.animationFrames.length,
      modelUpdateFps: this.modelRenders.length / Math.max(1, this.meshes.size),
      correctiveBatchMs: meanBatchTime,
      correctiveBackend: this.correctiveBackend,
    };
  }

  consume(message: Message): boolean {
    if (message.type === "BodyModelsViserAssetMessage") {
      this.addAsset(message as AssetMessage);
      return true;
    }
    if (message.type === "BodyModelsViserModelMessage") {
      this.addModel(message as ModelMessage);
      return true;
    }
    if (message.type === "BodyModelsViserIdentityMessage") {
      this.setIdentity(message as IdentityMessage);
      return true;
    }
    if (message.type === "BodyModelsViserPoseMessage") {
      this.setPose(message as PoseMessage);
      return true;
    }
    if (message.type === "BodyModelsViserTransformMessage") {
      this.setTransform(message as TransformMessage);
      return true;
    }
    if (message.type === "RemoveSceneNodeMessage") {
      this.remove(message as RemoveSceneNodeMessage);
    }
    return false;
  }

  private addAsset(message: AssetMessage): void {
    if (this.assets.has(message.asset_id)) {
      return;
    }
    let correctives: Correctives | null = null;
    if (message.corrective_format === "dense") {
      correctives = {
        format: "dense",
        values: null,
        scales: null,
        valueValues: message.corrective_values,
        scaleValues: message.corrective_scales,
      };
    } else if (message.corrective_format === "sparse") {
      correctives = {
        format: "sparse",
        values: null,
        scales: null,
        offsets: null,
        indices: null,
        valueValues: message.corrective_values,
        scaleValues: message.corrective_scales,
        offsetValues: message.corrective_offsets,
        indexValues: message.corrective_indices,
      };
    }
    const asset: AssetState = {
      id: message.asset_id,
      faces: message.faces,
      weightOffsets: this.copyToWasm(message.skin_weight_offsets),
      weightIndices: this.copyToWasm(message.skin_weight_indices),
      weightValues: this.copyToWasm(message.skin_weight_values),
      correctives,
      weightOffsetValues: message.skin_weight_offsets,
      weightIndexValues: message.skin_weight_indices,
      weightValueValues: message.skin_weight_values,
      meshes: new Set(),
      dirty: new Set(),
      scheduled: false,
      busy: false,
      disposeWhenIdle: false,
      gpu: null,
    };
    if (correctives !== null) {
      asset.gpu = this.createGpuAsset(asset).catch((error) => {
        this.warnGpuFallback(error);
        return null;
      });
    }
    this.assets.set(message.asset_id, asset);
  }

  private addModel(message: ModelMessage): void {
    const asset = this.assets.get(message.asset_id);
    if (asset === undefined) {
      throw new Error(`Body-model asset ${message.asset_id} has not been received.`);
    }
    const existing = this.meshes.get(message.name);
    if (existing !== undefined && existing.asset === asset) {
      this.updateModel(existing, message);
      return;
    }
    if (existing !== undefined) {
      this.remove({ type: "RemoveSceneNodeMessage", name: message.name });
    }
    const mesh: MeshState = {
      name: message.name,
      asset,
      restVertices: this.copyToWasm(message.rest_vertices),
      skinningTransforms: this.copyToWasm(message.skinning_transforms),
      poseCoefficients:
        message.pose_coefficients === null
          ? null
          : this.copyToWasm(message.pose_coefficients),
      poseOffsets: this.allocTyped(Float32Array, message.rest_vertices.length),
      globalRotation: this.copyToWasm(message.global_rotation),
      globalTranslation: this.copyToWasm(message.global_translation),
      outputVertices: this.allocTyped(Float32Array, message.rest_vertices.length),
      restValues: message.rest_vertices.slice(),
      transformValues: message.skinning_transforms.slice(),
      coefficientValues: message.pose_coefficients?.slice() ?? null,
      globalRotationValues: message.global_rotation.slice(),
      globalTranslationValues: message.global_translation.slice(),
      props: message.props,
      restVersion: 0,
      sceneStateSent: false,
      wxyz: message.wxyz,
      position: message.position,
      visible: message.visible,
    };
    asset.meshes.add(mesh);
    this.meshes.set(message.name, mesh);
    this.render(mesh);
  }

  private updateModel(mesh: MeshState, message: ModelMessage): void {
    this.updateBuffer(mesh.restVertices, message.rest_vertices);
    this.updateBuffer(mesh.skinningTransforms, message.skinning_transforms);
    this.updateBuffer(mesh.globalRotation, message.global_rotation);
    this.updateBuffer(mesh.globalTranslation, message.global_translation);
    if ((mesh.poseCoefficients === null) !== (message.pose_coefficients === null)) {
      throw new Error("Pose-corrective mode cannot change after a model is created.");
    }
    if (message.pose_coefficients !== null) {
      this.updateBuffer(mesh.poseCoefficients!, message.pose_coefficients);
    }
    mesh.restValues = message.rest_vertices.slice();
    mesh.transformValues = message.skinning_transforms.slice();
    mesh.coefficientValues = message.pose_coefficients?.slice() ?? null;
    mesh.globalRotationValues = message.global_rotation.slice();
    mesh.globalTranslationValues = message.global_translation.slice();
    mesh.props = message.props;
    mesh.restVersion++;
    this.render(mesh);
  }

  private setIdentity(message: IdentityMessage): void {
    const mesh = this.requireMesh(message.name);
    this.updateBuffer(mesh.restVertices, message.rest_vertices);
    mesh.restValues = message.rest_vertices.slice();
    mesh.restVersion++;
    this.updatePoseState(mesh, message.skinning_transforms, message.pose_coefficients);
  }

  private setPose(message: PoseMessage): void {
    const mesh = this.requireMesh(message.name);
    this.updatePoseState(mesh, message.skinning_transforms, message.pose_coefficients);
  }

  private updatePoseState(
    mesh: MeshState,
    transforms: Float32Array,
    coefficients: Float32Array | null,
  ): void {
    this.updateBuffer(mesh.skinningTransforms, transforms);
    mesh.transformValues = transforms.slice();
    if ((mesh.poseCoefficients === null) !== (coefficients === null)) {
      throw new Error("Pose-corrective mode cannot change after a model is created.");
    }
    if (coefficients !== null) {
      this.updateBuffer(mesh.poseCoefficients!, coefficients);
      mesh.coefficientValues = coefficients.slice();
    }
    this.render(mesh);
  }

  private setTransform(message: TransformMessage): void {
    const mesh = this.requireMesh(message.name);
    this.updateBuffer(mesh.globalRotation, message.global_rotation);
    this.updateBuffer(mesh.globalTranslation, message.global_translation);
    mesh.globalRotationValues = message.global_rotation.slice();
    mesh.globalTranslationValues = message.global_translation.slice();
    this.render(mesh);
  }

  private render(mesh: MeshState): void {
    if (mesh.poseCoefficients === null) {
      this.renderWasm(mesh);
      return;
    }
    mesh.asset.dirty.add(mesh);
    this.scheduleCorrectives(mesh.asset);
  }

  private scheduleCorrectives(asset: AssetState): void {
    if (asset.scheduled || asset.busy) {
      return;
    }
    asset.scheduled = true;
    queueMicrotask(() => {
      asset.scheduled = false;
      void this.renderCorrectiveBatch(asset);
    });
  }

  private async renderCorrectiveBatch(asset: AssetState): Promise<void> {
    if (asset.busy || asset.dirty.size === 0) {
      return;
    }
    asset.busy = true;
    // Keep a stable batch for every model sharing the asset. Rebuilding GPU
    // buffers for whichever subset happened to arrive before this animation
    // frame is slower than recomputing unchanged bodies, and can display one
    // logical server frame in several browser frames.
    const dirty = [...asset.dirty];
    const meshes = [...asset.meshes]
      .filter((mesh) => mesh.poseCoefficients !== null)
      .sort((a, b) => a.name.localeCompare(b.name));
    asset.dirty.clear();
    const started = performance.now();
    try {
      const gpu = asset.gpu === null ? null : await asset.gpu;
      if (gpu === null) {
        this.correctiveBackend = "wasm";
        this.renderWasmMeshes(dirty);
      } else {
        this.correctiveBackend = "webgpu";
        try {
          await this.renderGpuBatch(gpu, meshes);
        } catch (error) {
          this.warnGpuFallback(error);
          this.destroyGpuAsset(gpu);
          asset.gpu = Promise.resolve(null);
          this.correctiveBackend = "wasm";
          this.renderWasmMeshes(dirty);
        }
      }
    } finally {
      this.correctiveBatchTimes.push(performance.now() - started);
      asset.busy = false;
      if (asset.disposeWhenIdle) {
        this.freeAsset(asset);
        return;
      }
      if (asset.dirty.size !== 0) {
        this.scheduleCorrectives(asset);
      }
    }
  }

  private renderWasmMeshes(meshes: MeshState[]): void {
    for (const mesh of meshes) {
      if (this.meshes.get(mesh.name) === mesh) {
        this.renderWasm(mesh);
      }
    }
  }

  private renderWasm(mesh: MeshState): void {
    const wasm = this.requireWasm();
    if (mesh.poseCoefficients !== null) {
      const correctives = mesh.asset.correctives!;
      correctives.values ??= this.copyToWasm(correctives.valueValues);
      correctives.scales ??= this.copyToWasm(correctives.scaleValues);
      if (correctives.format === "dense") {
        wasm.compute_dense_pose_offsets(
          correctives.values.ptr,
          correctives.values.len,
          correctives.scales.ptr,
          correctives.scales.len,
          mesh.poseCoefficients.ptr,
          mesh.poseCoefficients.len,
          mesh.poseOffsets.ptr,
          mesh.poseOffsets.len,
        );
      } else {
        correctives.offsets ??= this.copyToWasm(correctives.offsetValues);
        correctives.indices ??= this.copyToWasm(correctives.indexValues);
        wasm.compute_sparse_pose_offsets(
          correctives.offsets.ptr,
          correctives.offsets.len,
          correctives.indices.ptr,
          correctives.indices.len,
          correctives.values.ptr,
          correctives.values.len,
          correctives.scales.ptr,
          correctives.scales.len,
          mesh.poseCoefficients.ptr,
          mesh.poseCoefficients.len,
          mesh.poseOffsets.ptr,
          mesh.poseOffsets.len,
        );
      }
    }
    wasm.forward_vertices_sparse(
      mesh.asset.weightOffsets.ptr,
      mesh.asset.weightOffsets.len,
      mesh.asset.weightIndices.ptr,
      mesh.asset.weightIndices.len,
      mesh.asset.weightValues.ptr,
      mesh.asset.weightValues.len,
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
    this.pushMesh(mesh, this.copyOutput(mesh.outputVertices));
  }

  private async renderGpuBatch(
    gpu: GpuAsset,
    meshes: MeshState[],
  ): Promise<void> {
    if (meshes.length === 0) {
      return;
    }
    const batch = this.prepareGpuBatch(gpu, meshes);
    const device = gpu.context.device;
    const firstMesh = meshes[0]!;
    const coefficientCount = firstMesh.coefficientValues!.length;
    const transformCount = firstMesh.transformValues.length;
    const coefficients = new Float32Array(meshes.length * coefficientCount);
    const transforms = new Float32Array(meshes.length * transformCount);
    const globals = new Float32Array(meshes.length * 6);
    for (let index = 0; index < meshes.length; index++) {
      const mesh = meshes[index]!;
      coefficients.set(mesh.coefficientValues!, index * coefficientCount);
      transforms.set(mesh.transformValues, index * transformCount);
      globals.set(mesh.globalRotationValues, index * 6);
      globals.set(mesh.globalTranslationValues, index * 6 + 3);
      if (batch.restVersions[index] !== mesh.restVersion) {
        device.queue.writeBuffer(
          batch.dynamic,
          (batch.restOffset + index * mesh.restValues.length) * 4,
          mesh.restValues,
        );
        batch.restVersions[index] = mesh.restVersion;
      }
    }
    device.queue.writeBuffer(batch.dynamic, batch.coefficientOffset * 4, coefficients);
    device.queue.writeBuffer(batch.dynamic, batch.transformOffset * 4, transforms);
    device.queue.writeBuffer(batch.dynamic, batch.globalOffset * 4, globals);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(gpu.context.pipeline);
    pass.setBindGroup(0, batch.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(firstMesh.restValues.length / 3 / 64), meshes.length);
    pass.end();
    encoder.copyBufferToBuffer(batch.output, 0, batch.readback, 0, batch.readback.size);
    device.queue.submit([encoder.finish()]);
    await batch.readback.mapAsync((globalThis as any).GPUMapMode.READ);
    const output = new Float32Array(batch.readback.getMappedRange()).slice();
    batch.readback.unmap();

    const coordinateCount = firstMesh.restValues.length;
    for (let index = 0; index < meshes.length; index++) {
      const mesh = meshes[index]!;
      if (this.meshes.get(mesh.name) !== mesh) {
        continue;
      }
      const start = index * coordinateCount;
      this.pushMesh(mesh, output.subarray(start, start + coordinateCount));
    }
  }

  private prepareGpuBatch(gpu: GpuAsset, meshes: MeshState[]): GpuBatch {
    if (
      gpu.batch !== null &&
      gpu.batch.meshes.length === meshes.length &&
      gpu.batch.meshes.every((mesh, index) => mesh === meshes[index])
    ) {
      return gpu.batch;
    }
    this.freeGpuBatch(gpu.batch);

    const device = gpu.context.device;
    const bodyCount = meshes.length;
    const firstMesh = meshes[0]!;
    const coefficientCount = firstMesh.coefficientValues!.length;
    const coordinateCount = firstMesh.restValues.length;
    const transformCount = firstMesh.transformValues.length;
    const coefficientOffset = 0;
    const restOffset = coefficientOffset + bodyCount * coefficientCount;
    const transformOffset = restOffset + bodyCount * coordinateCount;
    const globalOffset = transformOffset + bodyCount * transformCount;
    const dynamicLength = globalOffset + bodyCount * 6;
    const usage = (globalThis as any).GPUBufferUsage;
    const dynamic = device.createBuffer({
      size: dynamicLength * 4,
      usage: usage.STORAGE | usage.COPY_DST,
    });
    for (let index = 0; index < bodyCount; index++) {
      device.queue.writeBuffer(
        dynamic,
        (restOffset + index * coordinateCount) * 4,
        meshes[index]!.restValues,
      );
    }
    const output = device.createBuffer({
      size: bodyCount * coordinateCount * 4,
      usage: usage.STORAGE | usage.COPY_SRC,
    });
    const readback = device.createBuffer({
      size: bodyCount * coordinateCount * 4,
      usage: usage.COPY_DST | usage.MAP_READ,
    });
    const params = this.createGpuBuffer(
      device,
      new Uint32Array([
        coordinateCount / 3,
        coefficientCount,
        transformCount / 16,
        bodyCount,
        coefficientOffset,
        restOffset,
        transformOffset,
        globalOffset,
        firstMesh.asset.correctives!.format === "dense" ? 0 : 1,
        0,
        0,
        0,
      ]),
      usage.UNIFORM,
    );
    const bindGroup = device.createBindGroup({
      layout: gpu.context.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpu.correctiveValues } },
        { binding: 1, resource: { buffer: gpu.correctiveScales } },
        { binding: 2, resource: { buffer: gpu.correctiveOffsets } },
        { binding: 3, resource: { buffer: gpu.correctiveIndices } },
        { binding: 4, resource: { buffer: gpu.weightOffsets } },
        { binding: 5, resource: { buffer: gpu.weightIndices } },
        { binding: 6, resource: { buffer: gpu.weightValues } },
        { binding: 7, resource: { buffer: dynamic } },
        { binding: 8, resource: { buffer: output } },
        { binding: 9, resource: { buffer: params } },
      ],
    });
    gpu.batch = {
      meshes: [...meshes],
      restVersions: meshes.map((mesh) => mesh.restVersion),
      dynamic,
      output,
      readback,
      params,
      bindGroup,
      coefficientOffset,
      restOffset,
      transformOffset,
      globalOffset,
    };
    return gpu.batch;
  }

  private async createGpuAsset(asset: AssetState): Promise<GpuAsset | null> {
    const context = await this.getGpuContext();
    if (context === null) {
      return null;
    }
    const correctives = asset.correctives!;
    const usage = (globalThis as any).GPUBufferUsage;
    const correctiveBytes = new Uint8Array(
      Math.ceil(correctives.valueValues.byteLength / 4) * 4,
    );
    correctiveBytes.set(
      new Uint8Array(
        correctives.valueValues.buffer,
        correctives.valueValues.byteOffset,
        correctives.valueValues.byteLength,
      ),
    );
    return {
      context,
      correctiveValues: this.createGpuBuffer(
        context.device,
        correctiveBytes,
        usage.STORAGE,
      ),
      correctiveScales: this.createGpuBuffer(
        context.device,
        correctives.scaleValues,
        usage.STORAGE,
      ),
      correctiveOffsets: this.createGpuBuffer(
        context.device,
        correctives.format === "sparse"
          ? correctives.offsetValues
          : new Uint32Array([0]),
        usage.STORAGE,
      ),
      correctiveIndices: this.createGpuBuffer(
        context.device,
        correctives.format === "sparse"
          ? Uint32Array.from(correctives.indexValues)
          : new Uint32Array([0]),
        usage.STORAGE,
      ),
      weightOffsets: this.createGpuBuffer(
        context.device,
        asset.weightOffsetValues,
        usage.STORAGE,
      ),
      weightIndices: this.createGpuBuffer(
        context.device,
        Uint32Array.from(asset.weightIndexValues),
        usage.STORAGE,
      ),
      weightValues: this.createGpuBuffer(
        context.device,
        asset.weightValueValues,
        usage.STORAGE,
      ),
      batch: null,
    };
  }

  private getGpuContext(): Promise<GpuContext | null> {
    this.gpuContext ??= this.initializeGpu();
    return this.gpuContext;
  }

  private async initializeGpu(): Promise<GpuContext | null> {
    try {
      const gpu = (navigator as any).gpu;
      if (gpu === undefined) {
        return null;
      }
      const adapter = await gpu.requestAdapter();
      if (adapter === null) {
        return null;
      }
      const device = await adapter.requestDevice({
        requiredLimits: { maxStorageBuffersPerShaderStage: 9 },
      });
      const module = device.createShaderModule({ code: CORRECTIVE_SHADER });
      const pipeline = await device.createComputePipelineAsync({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
      return { device, pipeline };
    } catch (error) {
      this.warnGpuFallback(error);
      return null;
    }
  }

  private createGpuBuffer(device: GpuDevice, values: ArrayBufferView, usage: number): GpuBuffer {
    const buffer = device.createBuffer({
      size: Math.max(4, Math.ceil(values.byteLength / 4) * 4),
      usage,
      mappedAtCreation: true,
    });
    new Uint8Array(buffer.getMappedRange()).set(
      new Uint8Array(values.buffer, values.byteOffset, values.byteLength),
    );
    buffer.unmap();
    return buffer;
  }

  private pushMesh(mesh: MeshState, vertices: Float32Array): void {
    this.modelRenders.push(performance.now());
    const queue = this.getViewer().mutable.current.messageQueue;
    queue.push({
      type: "MeshMessage",
      name: mesh.name,
      props: { ...mesh.props, vertices, faces: mesh.asset.faces },
    });
    if (!mesh.sceneStateSent) {
      queue.push({ type: "SetOrientationMessage", name: mesh.name, wxyz: mesh.wxyz });
      queue.push({ type: "SetPositionMessage", name: mesh.name, position: mesh.position });
      queue.push({
        type: "SetSceneNodeVisibilityMessage",
        name: mesh.name,
        visible: mesh.visible,
      });
      mesh.sceneStateSent = true;
    }
  }

  private remove(message: RemoveSceneNodeMessage): void {
    const mesh = this.meshes.get(message.name);
    if (mesh === undefined) {
      return;
    }
    const asset = mesh.asset;
    asset.dirty.delete(mesh);
    asset.meshes.delete(mesh);
    this.freeMesh(mesh);
    this.meshes.delete(message.name);
    if (asset.meshes.size === 0) {
      this.assets.delete(asset.id);
      if (asset.busy) {
        asset.disposeWhenIdle = true;
      } else {
        this.freeAsset(asset);
      }
    }
  }

  private requireMesh(name: string): MeshState {
    const mesh = this.meshes.get(name);
    if (mesh === undefined) {
      throw new Error(`Body model ${name} has not been created.`);
    }
    return mesh;
  }

  private copyToWasm(array: ArrayBufferView): WasmBuffer {
    const elementSize = (array as ArrayBufferView & { BYTES_PER_ELEMENT?: number })
      .BYTES_PER_ELEMENT ?? 1;
    const buffer = this.allocBytes(array.byteLength, array.byteLength / elementSize);
    this.copyBytesIntoWasm(buffer, array);
    return buffer;
  }

  private allocTyped(
    type: { readonly BYTES_PER_ELEMENT: number },
    len: number,
  ): WasmBuffer {
    return this.allocBytes(len * type.BYTES_PER_ELEMENT, len);
  }

  private allocBytes(byteLen: number, len: number): WasmBuffer {
    const wasm = this.requireWasm();
    const ptr = wasm.alloc(byteLen);
    new Uint8Array(wasm.memory.buffer, ptr, byteLen).fill(0);
    return { ptr, len, byteLen };
  }

  private updateBuffer(buffer: WasmBuffer, array: ArrayBufferView): void {
    if (array.byteLength !== buffer.byteLen) {
      throw new Error(`Expected ${buffer.byteLen} bytes, received ${array.byteLength}.`);
    }
    this.copyBytesIntoWasm(buffer, array);
  }

  private copyBytesIntoWasm(buffer: WasmBuffer, array: ArrayBufferView): void {
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    new Uint8Array(this.requireWasm().memory.buffer, buffer.ptr, buffer.byteLen).set(bytes);
  }

  private copyOutput(buffer: WasmBuffer): Float32Array {
    const wasm = this.requireWasm();
    return new Float32Array(new Float32Array(wasm.memory.buffer, buffer.ptr, buffer.len));
  }

  private freeMesh(mesh: MeshState): void {
    this.freeBuffer(mesh.restVertices);
    this.freeBuffer(mesh.skinningTransforms);
    if (mesh.poseCoefficients !== null) {
      this.freeBuffer(mesh.poseCoefficients);
    }
    this.freeBuffer(mesh.poseOffsets);
    this.freeBuffer(mesh.globalRotation);
    this.freeBuffer(mesh.globalTranslation);
    this.freeBuffer(mesh.outputVertices);
  }

  private freeAsset(asset: AssetState): void {
    this.freeBuffer(asset.weightOffsets);
    this.freeBuffer(asset.weightIndices);
    this.freeBuffer(asset.weightValues);
    const correctives = asset.correctives;
    if (correctives !== null) {
      if (correctives.values !== null) {
        this.freeBuffer(correctives.values);
      }
      if (correctives.scales !== null) {
        this.freeBuffer(correctives.scales);
      }
      if (correctives.format === "sparse") {
        if (correctives.offsets !== null) {
          this.freeBuffer(correctives.offsets);
        }
        if (correctives.indices !== null) {
          this.freeBuffer(correctives.indices);
        }
      }
    }
    if (asset.gpu !== null) {
      void asset.gpu.then((gpu) => {
        if (gpu === null) {
          return;
        }
        this.destroyGpuAsset(gpu);
      });
    }
  }

  private destroyGpuAsset(gpu: GpuAsset): void {
    this.freeGpuBatch(gpu.batch);
    gpu.correctiveValues.destroy();
    gpu.correctiveScales.destroy();
    gpu.correctiveOffsets.destroy();
    gpu.correctiveIndices.destroy();
    gpu.weightOffsets.destroy();
    gpu.weightIndices.destroy();
    gpu.weightValues.destroy();
  }

  private warnGpuFallback(error: unknown): void {
    console.warn("body-models-viser: WebGPU failed; using WASM pose correctives.", error);
  }

  private freeGpuBatch(batch: GpuBatch | null): void {
    if (batch === null) {
      return;
    }
    batch.dynamic.destroy();
    batch.output.destroy();
    batch.readback.destroy();
    batch.params.destroy();
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
    const queue = this.getViewer().mutable.current.messageQueue;
    const push = queue.push.bind(queue);
    queue.push = (...messages: Message[]) => {
      const forwarded = messages.filter((message) => !this.consume(message));
      return push(...forwarded);
    };
  }

  private drainPreload(): void {
    const preload = getPreloadState();
    if (preload === undefined) {
      return;
    }
    const pending = preload.pending.splice(0);
    preload.restore();
    pending.forEach((message) => this.consume(message));
  }

  private trackAnimationFrames(): void {
    const tick = (now: number) => {
      this.animationFrames.push(now);
      this.pruneTimings(now);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  private pruneTimings(now: number): void {
    const cutoff = now - 1000;
    while (this.animationFrames[0] !== undefined && this.animationFrames[0] < cutoff) {
      this.animationFrames.shift();
    }
    while (this.modelRenders[0] !== undefined && this.modelRenders[0] < cutoff) {
      this.modelRenders.shift();
    }
    if (this.correctiveBatchTimes.length > 120) {
      this.correctiveBatchTimes.splice(0, this.correctiveBatchTimes.length - 120);
    }
  }
}

type ReactFiberNode = {
  memoizedProps?: { value?: Partial<ViewerLike> & Record<string, unknown> };
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
    if (fiber.child) stack.push(fiber.child);
    if (fiber.sibling) stack.push(fiber.sibling);
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

function getPreloadState(): PreloadState | undefined {
  return (window as unknown as Record<string, PreloadState | undefined>)[
    "__BODY_MODELS_VISER_PRELOAD__"
  ];
}

const runtime = new BodyModelsViserRuntime();

export function install(wasmBase64: string): void {
  runtime.install(wasmBase64);
}

export function ready(): void {
  runtime.ready();
}

export function stats(): Record<string, number | string> {
  return runtime.stats();
}
