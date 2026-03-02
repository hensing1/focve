using UnityEngine;

[ExecuteInEditMode]
public class SMPLInstantiator : SMPLInstantiatorBase
{
    public string posePath = "";
    public TextAsset[] poseAssets = null;

    public override GameObject[] Run()
    {
        // Only assign if not null
        poseAssets = Utility.LoadTextAssets(posePath) ?? poseAssets;

        GameObject[] instances = new GameObject[poseAssets.Length];

        for (int poseIndex = 0; poseIndex < poseAssets.Length; ++poseIndex)
        {
            var poseAsset = poseAssets[poseIndex];
            GameObject instance = Instantiate(prefab, Vector3.zero, Quaternion.identity);
            instance.transform.parent = this.transform;
            SkinnedMeshRenderer renderer = instance.GetComponentInChildren<SkinnedMeshRenderer>();
            SMPLBlendShapesUpdater blendShapesUpdater = renderer.gameObject.AddComponent<SMPLBlendShapesUpdater>();

            Transform camera = this.transform.Find(poseAsset.name[..5]);
            if (camera != null)
            {
                instance.transform.parent = camera;
            }

            FixPelvisTransform(renderer.bones);

            var pose = LoadSMPLPoseParams(poseAsset);

            Transform[] smplBones = Utility.GetSMPLBones(renderer.bones);
            pose.Apply(instance.transform, smplBones, blendShapesUpdater);

            // Save the created GameObject for the return value
            instances[poseIndex] = instance;
        }

        return instances;
    }
}
