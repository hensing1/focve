using UnityEngine;

public class SolarSystemController : MonoBehaviour
{
    public float sunRotationPeriod = 27.3f;
    public float sunEarthRotationPeriod = 365.0f;
    public float earthRotationPeriod = 1.0f;
    public float earthMoonRotationPeriod = 27.3f;

    /*
    // These would be realistic parameters:
    public float sunEarthDistance = 149000.6f; // thousand kilomenters
    public float earthMoonDistance = 385.0f; // thousand kilometers
    public float sunRadius = 695.7f; // thousand kilometers
    public float earthRadius = 6.371f; // thousand kilometers
    public float moonRadius = 1.738f; // thousand kilometers
    */

    // These are fictional parameters!
    public float sunEarthDistance = 10.0f;
    public float earthMoonDistance = 1.0f;
    public float sunRadius = 1.0f;
    public float earthRadius = 0.5f;
    public float moonRadius = 0.1f;

    /// <summary>
    /// Reset a transformation to the identity.
    /// </summary>
    /// <param name="t">The transform to reset</param>
    private void ResetToIdentity(Transform t)
    {
        t.localPosition = Vector3.zero;
        t.localRotation = Quaternion.identity;
        t.localScale = Vector3.one;
    }

    /// <summary>
    /// Set the sun, earth and moon transformation parameters for the given time point.
    /// </summary>
    /// <param name="time">Time passed since beginning of the simulation in days.</param>
    public void UpdateSolarSystem(float time)
    {
        // Get transforms of the planets
        Transform sun = GameObject.Find("Sun").transform;
        Transform earth = GameObject.Find("Earth").transform;
        Transform moon = GameObject.Find("Moon").transform;

        // Reset to identity
        ResetToIdentity(sun.transform);
        ResetToIdentity(earth.transform);
        ResetToIdentity(moon.transform);

        // TODO implement

    }

    // Update is called once per frame
    void Update()
    {
        // 1 real day per 1 game second.
        UpdateSolarSystem(Time.time);
    }
}
