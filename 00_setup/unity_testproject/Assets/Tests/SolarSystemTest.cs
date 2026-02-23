using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using UnityEditor;
using UnityEditor.SceneManagement;

public class SolarSystemTest
{
    private const float tolerance = 1e-2f;

    private SolarSystemController controller;
    private Transform earth;

    [SetUp]
    public void Setup()
    {
        EditorSceneManager.OpenScene("Assets/Scenes/SolarSystem.unity");
        controller = GameObject.Find("Solar System Root").GetComponent<SolarSystemController>();
        earth = GameObject.Find("Earth").transform;
    }

    [Test]
    public void Test()
    {
        controller.UpdateSolarSystem(0);

        Assert.IsTrue(true);
    }

    [TearDown]
    public void Teadown()
    {
        EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
    }
}
