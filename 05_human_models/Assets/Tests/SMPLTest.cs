using NUnit.Framework;
using UnityEngine;
using UnityEditor.SceneManagement;
using System.Text;

public class SMPLTest
{
    protected const float blendShapeTolerance = 5e-1f;
    protected const float positionTolerance = 5e-3f;

    protected GameObject root;
    protected DomeInstantiator domeInstantiator;
    protected SMPLInstantiator smplInstantiator;
    protected SMPLAverageInstantiator smplAverageInstantiator;
    protected SMPLAnimationInstantiator smplAnimationInstantiator;

    [SetUp]
    public void Setup()
    {
        EditorSceneManager.OpenScene("Assets/Scenes/TestScene.unity");
        root = GameObject.Find("Root");

        // Get all the relevant components
        domeInstantiator = root.GetComponent<DomeInstantiator>();
        smplInstantiator = root.GetComponent<SMPLInstantiator>();
        smplAverageInstantiator = root.GetComponent<SMPLAverageInstantiator>();
        smplAnimationInstantiator = root.GetComponent<SMPLAnimationInstantiator>();

        // Clear all existing children
        while (root.transform.childCount > 0)
            Object.DestroyImmediate(root.transform.GetChild(root.transform.childCount - 1).gameObject);

        // Make sure that we have a clean set of cameras
        domeInstantiator.Run();
    }

    /*
    /// <summary>
    /// This is just a sanity check.
    /// No points will be granted for this test.
    /// </summary>
    [Test]
    public void ValidScene()
    {
        Assert.That(root, Is.Not.Null);
        Assert.That(domeInstantiator, Is.Not.Null);
        Assert.That(domeInstantiator.calibrationDome, Is.Not.Null);
        Assert.That(smplInstantiator, Is.Not.Null);
        Assert.That(smplInstantiator.prefab, Is.Not.Null);
        Assert.That(smplAverageInstantiator, Is.Not.Null);
        Assert.That(smplAverageInstantiator.prefab, Is.Not.Null);
        Assert.That(smplAnimationInstantiator, Is.Not.Null);
        Assert.That(smplAnimationInstantiator.prefab, Is.Not.Null);
    }
    */

    /*
    /// <summary>
    /// This is just a sanity check.
    /// No points will be granted for this test.
    /// </summary>
    [Test]
    public void ValidateDome()
    {
        Assert.That(root.transform.childCount, Is.EqualTo(35));
    }
    */

    protected void VerifyModel(GameObject instance, float[] blendShapeValues, Vector3[] jointPositions)
    {
        SkinnedMeshRenderer meshRenderer = instance.GetComponentInChildren<SkinnedMeshRenderer>();

        // Verify **all** the blendShapeValues (!= betas) on the SkinnedMeshRenderer.

        int blendShapeCount = meshRenderer.sharedMesh.blendShapeCount;
        if (blendShapeValues == null)
        {
            StringBuilder builder = new();
            for (int i = 0; i < blendShapeCount; ++i)
            {
                if (i % 20 == 0) builder.Append("\n    ");
                builder.AppendFormat("{0:00.00f}, ", meshRenderer.GetBlendShapeWeight(i));
            }
            Debug.LogError($"float[] blendShapeValues = {{{builder}\n}};");
        }
        else
        {
            // NOTE: This implicitly validates joint angles!
            Assert.That(blendShapeCount, Is.EqualTo(blendShapeValues.Length));
            for (int i = 0; i < blendShapeValues.Length; ++i)
            {
                Assert.That(Mathf.Abs(meshRenderer.GetBlendShapeWeight(i) - blendShapeValues[i]), Is.LessThanOrEqualTo(blendShapeTolerance), $"blend shape {i}");
            }
        }

        // Verify all joint positions (!= angles) on the SkinnedMeshRenderer.

        Transform[] bones = Utility.GetSMPLBones(meshRenderer.bones);
        if (jointPositions == null)
        {
            StringBuilder builder = new();
            for (int i = 0; i < bones.Length; ++i)
            {
                builder.AppendFormat("    new({0:0.000f}, {1:0.000f}, {2:0.000f}),\n", bones[i].position.x, bones[i].position.y, bones[i].position.z);
            }
            Debug.LogError($"Vector3[] jointPositions = {{\n{builder}}};");
        }
        else
        {
            Assert.That(jointPositions.Length, Is.EqualTo(bones.Length));
            for (int i = 0; i < jointPositions.Length; ++i)
            {
                Assert.That((bones[i].position - jointPositions[i]).magnitude, Is.LessThan(positionTolerance));
            }
        }
    }

    [TearDown]
    public void Teadown()
    {
        EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
    }
}

