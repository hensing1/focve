using UnityEngine;
using System.Collections.Generic;
using System;

[ExecuteInEditMode]
public class SMPLBlendShapesUpdater : MonoBehaviour
{
	public readonly float[] shapeParms; // The pointer to the array is readonly, not the contents!

	public bool autoUpdateShape = false;
	public bool autoUpdatePose = false;

	private SkinnedMeshRenderer targetMeshRenderer;
	private string boneNamePrefix = "";

	private readonly float _shapeBlendsScale = 5.0f;
	private readonly int _numShapeParms = 10;

	SMPLBlendShapesUpdater()
	{
		shapeParms = new float[_numShapeParms];
	}

	void Awake()
	{
		EnsureMeshRenderer();
	}

	void Update()
	{
		EnsureMeshRenderer();

		if (autoUpdateShape)
			ApplyShapeBlendValues();
		if (autoUpdatePose)
			ApplyPoseBlendValues();
	}


	/*	Set the corrective pose blendshape values from current pose-parameters (joint angles)
	 */
	public void ApplyPoseBlendValues()
	{
		Transform[] _bones = targetMeshRenderer.bones;

		int doubledShapeParms = _numShapeParms * 2;

		for (int i = 0; i < _bones.Length; i++)
		{
			string boneName = _bones[i].name;

			// Remove f_avg/m_avg prefix
			boneName = boneName.Replace(boneNamePrefix, "");

			if (boneName == "root" || boneName == "Pelvis")
				continue;

			if (SMPLModifyBones.TryGetJointIndex(boneName, out int boneIndex))
			{
				float[] rot3x3 = Quat_to_3x3Mat(_bones[i].localRotation);

				// Can't use the 'boneIndex' value as-is from the _boneNameToJointIndex dict; 
				// The pose blendshapes have no values corresponding to Pelvis joint. 
				// Poseblendshapes start from hip-joint instead of Pelvis.
				// So we have to begin pose_blend indices from 'boneIndex-1'
				int idx = (boneIndex - 1) * 9 * 2;

				for (int mat_elem = 0; mat_elem < 9; mat_elem++)
				{
					float pos, neg;
					float theta = rot3x3[mat_elem];

					if (theta >= 0)
					{
						pos = theta;
						neg = 0.0f;
					}
					else
					{
						pos = 0.0f;
						neg = -theta;
					}

					int bl_idx_0 = doubledShapeParms + idx + (mat_elem * 2) + 0;
					int bl_idx_1 = doubledShapeParms + idx + (mat_elem * 2) + 1;
					targetMeshRenderer.SetBlendShapeWeight(bl_idx_0, pos * 100.0f);
					targetMeshRenderer.SetBlendShapeWeight(bl_idx_1, neg * 100.0f);
				}
			}
		}
	}


	private void EnsureMeshRenderer()
	{
		if (targetMeshRenderer == null)
		{
			targetMeshRenderer = GetComponent<SkinnedMeshRenderer>();
			foreach (Transform bone in targetMeshRenderer.bones)
			{
				if (bone.name.EndsWith("root"))
				{
					int index = bone.name.IndexOf("root");
					boneNamePrefix = bone.name[..index];
					break;
				}
			}
		}
	}

	/*  Convert Quaternions to rotation matrices
	 * 
	 * 	parms:
	 * 	- quat: 	the quaternion value to be converted to 3x3 rotation matrix
	 */
	private float[] Quat_to_3x3Mat(Quaternion quat)
	{
		// Converting quaternions from Unity's LHS to coordinate system 
		// RHS so that pose blendshapes get the correct values (because SMPL model's
		// pose-blendshapes were trained using a RHS coordinate system)
		float qx = quat.x * 1.0f;
		float qy = quat.y * -1.0f;
		float qz = quat.z * -1.0f;
		float qw = quat.w * 1.0f;

		float[] rot3x3 = new float[9];

		// Note: the -1 in indices 0, 4 & 8 are the rotation-np.eye(3) for pose-mapping of SMPL model
		rot3x3[0] = 1 - (2 * qy * qy) - (2 * qz * qz) - 1;
		rot3x3[1] = (2 * qx * qy) - (2 * qz * qw);
		rot3x3[2] = (2 * qx * qz) + (2 * qy * qw);

		rot3x3[3] = (2 * qx * qy) + (2 * qz * qw);
		rot3x3[4] = 1 - (2 * qx * qx) - (2 * qz * qz) - 1;
		rot3x3[5] = (2 * qy * qz) - (2 * qx * qw);

		rot3x3[6] = (2 * qx * qz) - (2 * qy * qw);
		rot3x3[7] = (2 * qy * qz) + (2 * qx * qw);
		rot3x3[8] = 1 - (2 * qx * qx) - (2 * qy * qy) - 1;

		// NOTE: this is rotation - identity!
		return rot3x3;
	}

	/*	Set the new shape-parameters (betas) to create a new body shape
	 */
	public void ApplyShapeBlendValues()
	{
		for (int i = 0; i < 10; i++)
		{
			float pos, neg;
			float beta = shapeParms[i] / _shapeBlendsScale;

			if (beta >= 0)
			{
				pos = beta;
				neg = 0.0f;
			}
			else
			{
				pos = 0.0f;
				neg = -beta;
			}

			targetMeshRenderer.SetBlendShapeWeight(i * 2 + 0, pos * 100.0f); // map [0, 1] space to [0, 100]
			targetMeshRenderer.SetBlendShapeWeight(i * 2 + 1, neg * 100.0f); // map [0, 1] space to [0, 100]
		}
	}

}
