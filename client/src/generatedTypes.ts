// AUTOMATICALLY GENERATED from schema/skinning.schema.json. Do not edit.

/**
 * @minItems 3
 * @maxItems 3
 */
export type Vec3 = [number, number, number];
/**
 * @minItems 4
 * @maxItems 4
 */
export type Mat4 = [
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number]
];

export interface SkinningInput {
  vertices: Vec3[];
  skinWeights: number[][];
  skinJoints: number[][];
  boneTransforms: Mat4[];
}
