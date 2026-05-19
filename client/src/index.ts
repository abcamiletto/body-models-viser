export type { Mat4, SkinningInput, Vec3 } from "./generatedTypes";

import type { Mat4, SkinningInput, Vec3 } from "./generatedTypes";

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
