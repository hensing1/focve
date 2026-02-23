using UnityEngine;

public class SolarSystemController : MonoBehaviour
{
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
        Transform earth = GameObject.Find("Earth").transform;

        // Reset to identity
        ResetToIdentity(earth.transform);
    }

    // Update is called once per frame
    void Update()
    {
        // 1 real day per 1 game second.
        UpdateSolarSystem(Time.time);
    }
}
