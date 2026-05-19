from __future__ import annotations

from pathlib import Path

HEADER = """// AUTOMATICALLY GENERATED body-models viser interfaces, from Python definitions.
// This file should not be manually modified.
"""

ALIASES = {
    "NumericArray": "number[] | number[][] | Float32Array | Float64Array",
    "BodyModelParams": "Record<string, NumericArray>",
    "Vec3": "[number, number, number]",
    "QuatWxyz": "[number, number, number, number]",
    "Face": "[number, number, number] | [number, number, number, number]",
    "Mat4": (
        "[[number, number, number, number], "
        "[number, number, number, number], "
        "[number, number, number, number], "
        "[number, number, number, number]]"
    ),
}

INTERFACES = {
    "BodyModelForwardOutput": {
        "skeleton": "readonly Mat4[]",
        "mesh": "readonly Vec3[]",
    },
    "BodyModelSceneFrameOptions": {
        "showAxes": "boolean",
    },
    "BodyModelSceneSkinnedMeshOptions": {
        "vertices": "readonly Vec3[]",
        "faces": "readonly [number, number, number][]",
        "boneWxyzs": "readonly QuatWxyz[]",
        "bonePositions": "readonly Vec3[]",
        "skinWeights": "readonly (readonly number[])[]",
        "color": "Vec3",
    },
    "ViserFrameHandle": {
        "name": "string",
        "wxyz": "QuatWxyz",
        "position": "Vec3",
        "visible": "boolean",
        "remove()": "void",
    },
    "ViserBoneHandle": {
        "wxyz": "QuatWxyz",
        "position": "Vec3",
    },
    "ViserSkinnedMeshHandle": {
        "vertices": "readonly Vec3[]",
        "bones": "readonly ViserBoneHandle[]",
        "remove()": "void",
    },
    "BodyModelScene": {
        "addFrame(name: string, options: BodyModelSceneFrameOptions)": "ViserFrameHandle",
        "addMeshSkinned(name: string, options: BodyModelSceneSkinnedMeshOptions)": "ViserSkinnedMeshHandle",
    },
}

BODY_MODEL_LIKE = """export interface BodyModelLike<TParams extends BodyModelParams = BodyModelParams> {
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
"""


def generate() -> str:
    lines = [HEADER]
    lines += [f"export type {name} = {value};" for name, value in ALIASES.items()]
    lines.append("")

    for name, fields in INTERFACES.items():
        lines.append(f"export interface {name} {{")
        for field, value in fields.items():
            lines.append(f"  {field}: {value};")
        lines.append("}")
        lines.append("")

    lines.append(BODY_MODEL_LIKE)
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "client" / "src" / "generatedTypes.ts"
    output.write_text(generate(), encoding="utf-8")


if __name__ == "__main__":
    main()
