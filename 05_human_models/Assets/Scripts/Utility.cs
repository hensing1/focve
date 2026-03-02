using UnityEngine;

public class Utility
{

    public static Transform[] GetSMPLBones(Transform[] bones)
    {
        string[] _boneNames = {
            // "Pelvis", NOPE!
            "L_Hip",
            "R_Hip",
            "Spine1",
            "L_Knee",
            "R_Knee",
            "Spine2",
            "L_Ankle",
            "R_Ankle",
            "Spine3",
            "L_Foot",
            "R_Foot",
            "Neck",
            "L_Collar",
            "R_Collar",
            "Head",
            "L_Shoulder",
            "R_Shoulder",
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",
            "L_Hand",
            "R_Hand"
            // "Jaw" /// Not part of Unity model!
        };

        Transform[] smplBones = new Transform[_boneNames.Length];
        foreach (var bone in bones)
        {
            for (int i = 0; i < _boneNames.Length; ++i)
            {
                if (bone.name.EndsWith(_boneNames[i]))
                {
                    smplBones[i] = bone;
                    break;
                }
            }
        }

        return smplBones;
    }

    public static Matrix4x4 DecodeMatrix(SimpleJSON.JSONNode node)
    {
        int rows;
        int cols;
        // Support for 3x3 and 4x4 matrices
        switch (node.Count)
        {
            case 9:
                rows = cols = 3;
                break;
            case 16:
                rows = cols = 4;
                break;
            default:
                throw new System.Exception($"Invalid number of entries in matrix: {node.Count}");
        }
        // Construct transformation matrix from json array
        Matrix4x4 mat = Matrix4x4.identity;
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                mat[r, c] = node[r * cols + c].AsFloat;
            }
        }
        return mat;
    }

    public static Quaternion DecodeRotation(SimpleJSON.JSONNode node)
    {
        // Construct transformation matrix from json array
        Matrix4x4 mat = DecodeMatrix(node);

        // Get the rotation quaternion.
        // NOTE: Unity uses LHS coordinate system, SMPL assumes RHS coordinate system.
        // The x-axis is flipped.
        // diag(-1, 1, 1) * R * diag(-1, 1, 1) leaves the y-axis and z-axis of the rotation vector flipped.
        Quaternion quat = mat.rotation;
        quat.x *= 1;
        quat.y *= -1;
        quat.z *= -1;
        quat.w *= 1;

        return quat;
    }

    public static Vector3 DecodeTranslation(SimpleJSON.JSONNode node)
    {
        // NOTE: Unity uses LHS coordinate system, SMPL assumes RHS coordinate system.
        // The x-axis is flipped.
        return new Vector3(-node[0].AsFloat, node[1].AsFloat, node[2].AsFloat);
    }

    public static TextAsset[] LoadTextAssets(string path)
    {
        if (path == "")
            return null;

        Object[] assets = Resources.LoadAll(path, typeof(TextAsset));
        TextAsset[] textAssets = new TextAsset[assets.Length];
        for (int i = 0; i < assets.Length; ++i)
        {
            textAssets[i] = assets[i] as TextAsset;
        }
        return textAssets;
    }

}
