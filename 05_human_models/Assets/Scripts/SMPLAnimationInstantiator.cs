using UnityEngine;

[ExecuteInEditMode]
public class SMPLAnimationInstantiator : SMPLInstantiatorBase
{
    public string posePath = "";
    public TextAsset[] poseAssets = null;

    public override GameObject[] Run()
    {
        // Only assign if not null
        poseAssets = Utility.LoadTextAssets(posePath) ?? poseAssets;

        GameObject instance = Instantiate(prefab, Vector3.zero, Quaternion.identity);
        instance.transform.parent = this.transform;
        var targetMeshRenderer = instance.GetComponentInChildren<SkinnedMeshRenderer>();
        targetMeshRenderer.gameObject.AddComponent<SMPLBlendShapesUpdater>();
        var interpolator = instance.AddComponent<SMPLInterpolator>();

        Transform camera = this.transform.Find(poseAssets[0].name[..5]);
        if (camera != null)
        {
            instance.transform.parent = camera;
        }

        FixPelvisTransform(targetMeshRenderer.bones);

        interpolator.poseSequence = new SMPLPose[poseAssets.Length];
        for (int i = 0; i < poseAssets.Length; ++i)
        {
            interpolator.poseSequence[i] = LoadSMPLPoseParams(poseAssets[i]);
            interpolator.poseSequence[i].rootRotation.ToAngleAxis(out float angle, out Vector3 axis);
        }

        GameObject[] instances = { instance };
        return instances;
    }
}
