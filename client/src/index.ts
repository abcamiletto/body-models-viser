export type { Mat4, SkinningInput, Vec3 } from "./generatedTypes";

import type { Mat4, SkinningInput, Vec3 } from "./generatedTypes";

type RuntimeMessage = {
  type: string;
  [key: string]: unknown;
};

type BodyModelMeshMessage = {
  type: "BodyModelMeshMessage";
  name: string;
  vertices: Vec3[];
  faces: Vec3[];
  skinWeights: number[][];
  skinJoints: number[][];
  boneTransforms: Mat4[];
  color: [number, number, number];
  wireframe: boolean;
  opacity: number | null;
  flatShading: boolean;
  side: "front" | "back" | "double";
  material: "standard" | "toon3" | "toon5";
  scale: number | [number, number, number];
  castShadow: boolean;
  receiveShadow: boolean | number;
};

type BodyModelPoseMessage = {
  type: "BodyModelPoseMessage";
  name: string;
  vertices: Vec3[];
  boneTransforms: Mat4[];
};

type ViewerLike = {
  mutable: {
    current: {
      messageQueue: RuntimeMessage[];
    };
  };
};

type BodyModelState = {
  create: BodyModelMeshMessage;
};

export function skinVertices(input: SkinningInput): Vec3[] {
  const out: Vec3[] = [];
  for (let vertex = 0; vertex < input.vertices.length; vertex++) {
    const weights = input.skinWeights[vertex];
    const joints = input.skinJoints[vertex];
    if (weights === undefined || joints === undefined || weights.length !== joints.length) {
      throw new Error(`Invalid skinning weights for vertex ${vertex}.`);
    }

    let x = 0.0;
    let y = 0.0;
    let z = 0.0;
    for (let slot = 0; slot < weights.length; slot++) {
      const weight = weights[slot]!;
      const point = input.vertices[vertex]!;
      const transform = input.boneTransforms[joints[slot]!];
      if (transform === undefined) {
        throw new Error(`Vertex ${vertex} references missing bone ${joints[slot]}.`);
      }
      x += weight * (transform[0][0] * point[0] + transform[0][1] * point[1] + transform[0][2] * point[2] + transform[0][3]);
      y += weight * (transform[1][0] * point[0] + transform[1][1] * point[1] + transform[1][2] * point[2] + transform[1][3]);
      z += weight * (transform[2][0] * point[0] + transform[2][1] * point[1] + transform[2][2] * point[2] + transform[2][3]);
    }
    out.push([x, y, z]);
  }
  return out;
}

export function installViserRuntime(): void {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return;
  }

  const windowRef = window as Window & {
    __BODY_MODELS_VISER__?: BodyModelsViserRuntime;
  };
  windowRef.__BODY_MODELS_VISER__?.dispose();
  windowRef.__BODY_MODELS_VISER__ = new BodyModelsViserRuntime();
}

class BodyModelsViserRuntime {
  private viewer: ViewerLike | null = null;
  private states = new Map<string, BodyModelState>();
  private undoQueueIngress: (() => void) | null = null;
  private disposed = false;

  constructor() {
    this.installWhenReady();
  }

  dispose(): void {
    this.disposed = true;
    this.states.clear();
    this.undoQueueIngress?.();
    this.undoQueueIngress = null;
    this.viewer = null;
  }

  private installWhenReady(): void {
    if (this.disposed) {
      return;
    }
    try {
      const queue = this.getViewer().mutable.current.messageQueue;
      const push = queue.push.bind(queue);
      const wrappedPush = (...messages: RuntimeMessage[]): number => {
        const forwarded: RuntimeMessage[] = [];
        for (const message of messages) {
          const handled = this.handleMessage(message, forwarded);
          if (!handled) {
            forwarded.push(message);
          }
        }
        return push(...forwarded);
      };
      queue.push = wrappedPush;
      this.undoQueueIngress = () => {
        if (queue.push === wrappedPush) {
          queue.push = push;
        }
      };
      this.drainPending(queue);
    } catch {
      window.requestAnimationFrame(() => this.installWhenReady());
    }
  }

  private getViewer(): ViewerLike {
    if (!this.viewer) {
      this.viewer = findViewer();
    }
    return this.viewer;
  }

  private drainPending(queue: RuntimeMessage[]): void {
    const forwarded: RuntimeMessage[] = [];
    for (const message of queue.splice(0)) {
      if (!this.handleMessage(message, forwarded)) {
        forwarded.push(message);
      }
    }
    queue.push(...forwarded);
  }

  private handleMessage(message: RuntimeMessage, out: RuntimeMessage[]): boolean {
    if (isBodyModelMeshMessage(message)) {
      this.states.set(message.name, { create: message });
      out.push(this.meshMessage(message, message.vertices, message.boneTransforms));
      return true;
    }
    if (isBodyModelPoseMessage(message)) {
      const state = this.states.get(message.name);
      if (!state) {
        throw new Error(`Body model ${message.name} has not been created.`);
      }
      state.create.vertices = message.vertices;
      state.create.boneTransforms = message.boneTransforms;
      out.push(this.meshMessage(state.create, message.vertices, message.boneTransforms));
      return true;
    }
    return false;
  }

  private meshMessage(
    message: BodyModelMeshMessage,
    vertices: Vec3[],
    boneTransforms: Mat4[],
  ): RuntimeMessage {
    const skinned = skinVertices({
      vertices,
      skinWeights: message.skinWeights,
      skinJoints: message.skinJoints,
      boneTransforms,
    });
    return {
      type: "MeshMessage",
      name: message.name,
      props: {
        vertices: flattenFloat32(skinned),
        faces: flattenUint32(message.faces),
        color: message.color,
        wireframe: message.wireframe,
        opacity: message.opacity,
        flat_shading: message.flatShading,
        side: message.side,
        material: message.material,
        scale: message.scale,
        cast_shadow: message.castShadow,
        receive_shadow: message.receiveShadow,
      },
    };
  }
}

function flattenFloat32(values: Vec3[]): Float32Array {
  const out = new Float32Array(values.length * 3);
  for (let i = 0; i < values.length; i++) {
    const value = values[i]!;
    out[3 * i] = value[0];
    out[3 * i + 1] = value[1];
    out[3 * i + 2] = value[2];
  }
  return out;
}

function flattenUint32(values: Vec3[]): Uint32Array {
  const out = new Uint32Array(values.length * 3);
  for (let i = 0; i < values.length; i++) {
    const value = values[i]!;
    out[3 * i] = value[0];
    out[3 * i + 1] = value[1];
    out[3 * i + 2] = value[2];
  }
  return out;
}

function isBodyModelMeshMessage(message: RuntimeMessage): message is BodyModelMeshMessage {
  return message.type === "BodyModelMeshMessage";
}

function isBodyModelPoseMessage(message: RuntimeMessage): message is BodyModelPoseMessage {
  return message.type === "BodyModelPoseMessage";
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
  const key = root
    ? Object.keys(root).find((candidate) => candidate.startsWith("__reactContainer$"))
    : undefined;
  const reactRoot = key ? (root![key] as ReactFiberNode) : null;
  const seen = new Set<unknown>();
  const stack = reactRoot ? [reactRoot] : [];

  while (stack.length > 0) {
    const fiber = stack.pop();
    if (!fiber || seen.has(fiber)) {
      continue;
    }
    seen.add(fiber);
    const value = fiber.memoizedProps?.value;
    if (isViewerLike(value)) {
      return value;
    }
    if (fiber.child) {
      stack.push(fiber.child);
    }
    if (fiber.sibling) {
      stack.push(fiber.sibling);
    }
  }
  throw new Error("[body-models-viser] Could not locate the viser viewer.");
}

function isViewerLike(value: unknown): value is ViewerLike {
  if (!value || typeof value !== "object") {
    return false;
  }
  const candidate = value as Partial<ViewerLike>;
  return Array.isArray(candidate.mutable?.current?.messageQueue);
}

installViserRuntime();
