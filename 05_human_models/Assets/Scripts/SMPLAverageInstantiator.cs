using UnityEngine;

[ExecuteInEditMode]
public class SMPLAverageInstantiator : SMPLInstantiatorBase
{
    public string posePath = "";
    public TextAsset[] poseAssets = null;

    public override GameObject[] Run()
    {
        // Only assign if not null
        poseAssets = Utility.LoadTextAssets(posePath) ?? poseAssets;

        // Compute the mean pose....
        var meanPose = new SMPLPose();
        // For rotations, we can't directly accumulate into the Quaternions!
        Vector4 meanRootRotationAcc = Vector4.zero;
        Vector4[] meanJointRotationsAcc = new Vector4[SMPLPose.NUM_JOINT_PARAMS];

        // Load all the poses
        for (int i = 0; i < poseAssets.Length; ++i)
        {
            var pose = LoadSMPLPoseParams(poseAssets[i]);

            // Query the camera and apply its transform to the root transform of the pose!
            Transform camera = this.transform.Find(poseAssets[i].name[..5]);
            if (camera != null)
            {
                // Apply camera transform to root pose transform
                camera.GetLocalPositionAndRotation(out Vector3 camPosition, out Quaternion camRotation);
                pose.rootPosition = camRotation * pose.rootPosition + camPosition;
                pose.rootRotation = camRotation * pose.rootRotation;
            }

            // TODO implement

        }

        // TODO implement

        GameObject instance = Instantiate(prefab, Vector3.zero, Quaternion.identity);
        instance.transform.parent = this.transform;
        SkinnedMeshRenderer renderer = instance.GetComponentInChildren<SkinnedMeshRenderer>();
        SMPLBlendShapesUpdater blendShapesUpdater = renderer.gameObject.AddComponent<SMPLBlendShapesUpdater>();

        FixPelvisTransform(renderer.bones);

        Transform[] smplBones = Utility.GetSMPLBones(renderer.bones);
        meanPose.Apply(instance.transform, smplBones, blendShapesUpdater);

        GameObject[] instances = { instance };
        return instances;
    }
}
