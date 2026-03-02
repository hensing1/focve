using UnityEngine;
using System.Collections.Generic;

public class SMPLModifyBones {

	private readonly SkinnedMeshRenderer targetRenderer;

	public Transform[] Bones { get; private set; } = null;
    private Vector3[] _bonePositions;
	private Vector3[] _bonePositionsBackup = null;
    private Quaternion[] _boneRotationsBackup = null;

    public Transform Pelvis { get; private set; }

    public string BoneNamePrefix { get; private set; } = "";

    private static readonly Dictionary<string, int> boneNameToJointIndex = new()
    {
        { "Pelvis", 0 },
        { "L_Hip", 1 },
        { "R_Hip", 2 },
        { "Spine1", 3 },
        { "L_Knee", 4 },
        { "R_Knee", 5 },
        { "Spine2", 6 },
        { "L_Ankle", 7 },
        { "R_Ankle", 8 },
        { "Spine3", 9 },
        { "L_Foot", 10 },
        { "R_Foot", 11 },
        { "Neck", 12 },
        { "L_Collar", 13 },
        { "R_Collar", 14 },
        { "Head", 15 },
        { "L_Shoulder", 16 },
        { "R_Shoulder", 17 },
        { "L_Elbow", 18 },
        { "R_Elbow", 19 },
        { "L_Wrist", 20 },
        { "R_Wrist", 21 },
        { "L_Hand", 22 },
        { "R_Hand", 23 }
    };
    public static bool TryGetJointIndex(string boneName, out int index)
    {
        return boneNameToJointIndex.TryGetValue(boneName, out index);
    }

    private bool _initialized = false;

	public SMPLModifyBones(SkinnedMeshRenderer tr)
    {
		targetRenderer = tr;
    }

	// Use this for initialization
	public void Init()
    {
        Deinit();

		if (targetRenderer == null)
        {
            throw new System.ArgumentNullException("ERROR: The script should be added to the 'SkinnedMeshRenderer Object");
        }

		Bones = targetRenderer.bones;
        _bonePositions = new Vector3[Bones.Length];
        _bonePositionsBackup = new Vector3[Bones.Length];
        _boneRotationsBackup = new Quaternion[Bones.Length];
        BackupBones();

        // Determine bone name prefix
        foreach (Transform bone in Bones)
        {
            if (bone.name.EndsWith("root"))
            {
                int index = bone.name.IndexOf("root");
                BoneNamePrefix = bone.name.Substring(0, index);
                break;
            }
        }

        // Determine pelvis node
        foreach (Transform bone in Bones)
        {
            if (bone.name.EndsWith("Pelvis"))
            {
                Pelvis = bone;
                break;
            }
        }

        Debug.Log("INFO: Bone name prefix: '" + BoneNamePrefix + "'");
        _initialized = true;
    }

	public void Deinit()
    {
        if (!_initialized)
            return;

        RestoreBones();
        UpdateBindPose();
        _initialized = false;
    }

    public bool UpdateBonePositions(Vector3[] newPositions)
    {
        if (!_initialized)
            return false;

        // Save bone angles locally...
        var localBoneLocalRotationsBackup = new Quaternion[Bones.Length];
        for (int i = 0; i < Bones.Length; ++i)
        {
            localBoneLocalRotationsBackup[i] = Bones[i].localRotation;
        }

        // Restore initial T-pose
        RestoreBones();

        for (int i = 0; i < Bones.Length; i++)
        {
            string boneName = Bones[i].name;

            // Remove f_avg/m_avg prefix
            boneName = boneName.Replace(BoneNamePrefix, "");

            if (boneName == "root")
                continue;

            Transform avatarTransform = targetRenderer.transform.parent;
            if (TryGetJointIndex(boneName, out int index))
            {
                // Incoming new positions from joint calculation are centered at origin in world space
                // Transform to avatar position+orientation for correct world space position
                _bonePositions[i] = avatarTransform.TransformPoint(newPositions[index]);
                Bones[i].position = _bonePositions[i];
            }
            else
            {
                Debug.LogError("ERROR: No joint index for given bone name: " + boneName);
            }
        }

        UpdateBindPose();

        // Restore bone angles locally...
        for (int i = 0; i < Bones.Length; ++i)
        {
            Bones[i].localRotation = localBoneLocalRotationsBackup[i];
        }

        return true;
	}

	public bool UpdateBoneAngles(float[][] pose, float[] trans)
	{	
        if (!_initialized)
            return false;

		Quaternion quat;
		int pelvisIndex = -1;

		for (int i=0; i < Bones.Length; i++)
		{
			string boneName = Bones[i].name;

			// Remove f_avg/m_avg prefix
			boneName = boneName.Replace(BoneNamePrefix, "");

			if (boneName == "root")
				continue;

			if (boneName == "Pelvis")
				pelvisIndex = i;
			
			if (TryGetJointIndex(boneName, out int index))
			{
				quat.x = pose[index][0];
				quat.y = pose[index][1];
				quat.z = pose[index][2];
				quat.w = pose[index][3];

				/*	Quaternions */
                Bones[i].localRotation = quat;
			}
			else
			{
				Debug.LogError("ERROR: No joint index for given bone name: " + boneName);
			}
		}
			
		Bones[pelvisIndex].localPosition = new Vector3(trans[0], trans[1], trans[2]);
		return true;
	}

	public bool UpdateBoneAngles(Quaternion[] pose, Vector3 trans)
	{	
        if (!_initialized)
            return false;

		int pelvisIndex = -1;

		for (int i=0; i < Bones.Length; i++)
		{
			string boneName = Bones[i].name;

			// Remove f_avg/m_avg prefix
			boneName = boneName.Replace(BoneNamePrefix, "");

			if (boneName == "root")
				continue;

			if (boneName == "Pelvis")
				pelvisIndex = i;
			
			if (TryGetJointIndex(boneName, out int index))
			{
				/*	Quaternions */
                Bones[i].localRotation = pose[index];
			}
			else
			{
				Debug.LogError("ERROR: No joint index for given bone name: " + boneName);
			}
		}

		Bones[pelvisIndex].localPosition = trans;
		return true;
	}



    private void BackupBones()
	{
        // Save position + rotation of bones
		for (int i = 0; i < Bones.Length; i++)
        {
            _bonePositionsBackup[i] = Bones[i].position;
            _boneRotationsBackup[i] = Bones[i].rotation;
        }
	}	

	private void RestoreBones()
	{
		// Restore saved bones
		for (int i = 0; i < Bones.Length; i++)
		{
			Bones[i].position = _bonePositionsBackup[i];
			Bones[i].rotation = _boneRotationsBackup[i];
		}
	}	

	private void UpdateBindPose()
	{
		Matrix4x4[] bindPoses = targetRenderer.sharedMesh.bindposes;

        Transform avatarRootTransform = targetRenderer.transform.parent;

		for (int i = 0; i < Bones.Length; i++)
		{
	        // The bind pose is bone's inverse transformation matrix.
	        // Make this matrix relative to the avatar root so that we can move the root game object around freely.            
            bindPoses[i] = Bones[i].worldToLocalMatrix * avatarRootTransform.localToWorldMatrix;
		}

		///targetRenderer.bones = Bones;
		targetRenderer.sharedMesh.bindposes = bindPoses;
	}

}
