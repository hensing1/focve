using UnityEngine;

public class SMPLPose : ScriptableObject
{
    public const int NUM_SHAPE_PARAMS = 10;
    public const int NUM_JOINT_PARAMS = 23;
    public float[] shapeParams = new float[NUM_SHAPE_PARAMS];
    public Quaternion[] jointRotations = new Quaternion[NUM_JOINT_PARAMS];
    public Vector3 rootPosition;
    public Quaternion rootRotation;

    public void Apply(Transform root, Transform[] smplBones, SMPLBlendShapesUpdater blendShapesUpdater)
    {
        // Set the bone/joint rotations
        for (int i = 0; i < NUM_JOINT_PARAMS; ++i)
        {
            if (smplBones[i] == null)
                continue;
            smplBones[i].localRotation = jointRotations[i];
        }

        // Set the shape parameters
        for (int i = 0; i < NUM_SHAPE_PARAMS; ++i)
        {
            blendShapesUpdater.shapeParms[i] = shapeParams[i];
        }

        blendShapesUpdater.ApplyShapeBlendValues();
        blendShapesUpdater.ApplyPoseBlendValues();

        root.SetLocalPositionAndRotation(rootPosition, rootRotation);
    }
}
