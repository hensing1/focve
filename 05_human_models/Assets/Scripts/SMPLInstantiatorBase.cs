using UnityEngine;
using EasyButtons;

public class SMPLInstantiatorBase : MonoBehaviour
{
    public GameObject prefab;

    [Button]
    public virtual GameObject[] Run()
    {
        return null;
    }

    // The origin of the FBX model used in Unity is located at the floor.
    // The SMPL pose parameters assume that the pelvis joint is the origin of the model.
    // Therefore, we simply reset the transform of the pelvis Transform.
    public static void FixPelvisTransform(Transform[] bones)
    {
        foreach (var bone in bones)
        {
            if (bone.name.EndsWith("Pelvis"))
            {
                bone.SetLocalPositionAndRotation(Vector3.zero, Quaternion.identity);
                break;
            }
        }
    }

    // Decode the json file into an instance of the SMPLPose class, storing all the relevant parameters and joint angles as specified in the json file.
    public static SMPLPose LoadSMPLPoseParams(TextAsset jsonAsset)
    {
        var poseParams = ScriptableObject.CreateInstance<SMPLPose>();

        SimpleJSON.JSONNode node = SimpleJSON.JSON.Parse(jsonAsset.text);

        // TODO implement

        // After "import" we have: x-axis=left, y-axis=down, z-axis=forward
        // In Unity we want: x-axis=right, y-axis=up, z-axis=forward
        // Flip x-axis and y-axis == 180° rotation around z-axis
        Quaternion smpl_imported_to_unity = Quaternion.Euler(0, 0, 180);
        poseParams.rootRotation = smpl_imported_to_unity * poseParams.rootRotation;
        poseParams.rootPosition = smpl_imported_to_unity * poseParams.rootPosition;

        return poseParams;
    }

}
