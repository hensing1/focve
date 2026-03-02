using UnityEngine;

[ExecuteInEditMode]
public class SMPLInterpolator : MonoBehaviour
{
    public SMPLPose[] poseSequence;
    private SMPLPose interpolatedPose;

    public float fps = 30;
    public float time = 0;

    private SkinnedMeshRenderer targetMeshRenderer;
    private SMPLBlendShapesUpdater blendShapesUpdater;

    public void AssignPose()
    {
        // Set the bone/joint rotations
        Transform[] smplBones = Utility.GetSMPLBones(targetMeshRenderer.bones);
        for (int i = 0; i < smplBones.Length; ++i)
        {
            if (smplBones[i] == null)
                continue;
            smplBones[i].localRotation = interpolatedPose.jointRotations[i];
        }

        // Set the shape parameters
        for (int i = 0; i < interpolatedPose.shapeParams.Length; ++i)
        {
            blendShapesUpdater.shapeParms[i] = interpolatedPose.shapeParams[i];
        }

        // Update blend shapes
        blendShapesUpdater.ApplyShapeBlendValues();
        blendShapesUpdater.ApplyPoseBlendValues();

        // Apply root transform
        transform.SetLocalPositionAndRotation(interpolatedPose.rootPosition, interpolatedPose.rootRotation);
    }

    public void Update()
    {
        // Make sure that variables are initialized!
        // This is fancy syntax for if null then assign...
        targetMeshRenderer ??= GetComponentInChildren<SkinnedMeshRenderer>();
        blendShapesUpdater ??= GetComponentInChildren<SMPLBlendShapesUpdater>();
        interpolatedPose ??= ScriptableObject.CreateInstance<SMPLPose>();

        InterpolatePose();

        Transform[] smplBones = Utility.GetSMPLBones(targetMeshRenderer.bones);
        interpolatedPose.Apply(transform, smplBones, blendShapesUpdater);
    }

    public void InterpolatePose()
    {
        if (poseSequence?.Length == 0)
            return;

        // Get the adjacent keyframes and interpolation factor
        float poseIndexF = time * fps;
        poseIndexF = Mathf.Clamp(poseIndexF, 0, poseSequence.Length - 1);

        int poseIndex0 = Mathf.Clamp(Mathf.FloorToInt(poseIndexF), 0, poseSequence.Length - 2);
        int poseIndex1 = Mathf.Min(poseIndex0 + 1, poseSequence.Length - 1);
        SMPLPose pose0 = poseSequence[poseIndex0];
        SMPLPose pose1 = poseSequence[poseIndex1];
        float alpha = poseIndexF - poseIndex0;

        // TODO implement

    }
}
