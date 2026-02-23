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
        sun.localScale = sunRadius * Vector3.one;
        earth.localScale = earthRadius * Vector3.one;
        moon.localScale = moonRadius * Vector3.one;
        
        sun.Rotate(0, -360 * time / sunRotationPeriod, 0);

        earth.Translate(sunEarthDistance, 0, 0);
        earth.RotateAround(Vector3.zero, Vector3.up, -360 * time / sunEarthRotationPeriod);
        earth.RotateAround(earth.localPosition, Vector3.up, -360 * time / earthRotationPeriod);

        moon.Translate(sunEarthDistance, 0, 0);
        moon.RotateAround(Vector3.zero, Vector3.up, -360 * time / sunEarthRotationPeriod);
        moon.Translate(earthMoonDistance, 0, 0);
        moon.RotateAround(earth.localPosition, Vector3.up, -360 * time / earthMoonRotationPeriod);

        // sun.localRotation = Quaternion.Euler(new Vector3(0, -360 * time / sunRotationPeriod, 0));
        // earth.localRotation = Quaternion.Euler(new Vector3(0, -360 * time / earthRotationPeriod, 0));
        //
        // Matrix4x4 earthRotation = Matrix4x4.Rotate(Quaternion.Euler(0, -360 * time / sunEarthRotationPeriod, 0));
        // earth.localPosition = earthRotation.MultiplyPoint(new Vector3(sunEarthDistance, 0, 0));
        //
        // Matrix4x4 moonRotation = Matrix4x4.Rotate(Quaternion.Euler(0, -360 * time / earthMoonRotationPeriod, 0));
        // Vector3 earthToMoon = moonRotation.MultiplyPoint(new Vector3(earthMoonDistance, 0, 0));
        // moon.localPosition = earth.localPosition + earthToMoon;

    }

    // Update is called once per frame
    void Update()
    {
        // 1 real day per 1 game second.
        UpdateSolarSystem(Time.time);
    }
}
