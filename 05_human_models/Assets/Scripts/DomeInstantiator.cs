using UnityEngine;
using EasyButtons;

[ExecuteInEditMode]
public class DomeInstantiator : MonoBehaviour
{
    public TextAsset calibrationDome;

    public Vector3 domeToWorldRotation = new(0, 0, -90);
    public bool uprightCameras = true;


    [Button]
    public void Run()
    {
        SimpleJSON.JSONNode node = SimpleJSON.JSON.Parse(calibrationDome.text);
        SimpleJSON.JSONNode cameras_node = node["cameras"];
        for (int i = 0; i < cameras_node.Count; ++i)
        {
            SimpleJSON.JSONNode camera_node = cameras_node[i];

            var gameObject = new GameObject
            {
                name = camera_node["camera_id"]
            };
            gameObject.transform.parent = this.transform;
            var camera = gameObject.AddComponent<Camera>();
            camera.enabled = false;

            Matrix4x4 view_mat = Utility.DecodeMatrix(camera_node["extrinsics"]["view_matrix"]);
            // In Unity we specify "local to world" transformations,
            // The view matrix specifies a world to local transformation...
            Matrix4x4 inv_view_mat = view_mat.inverse;

            // The (inverse) view matrix is applied in the OpenCV coordinate system
            Matrix4x4 opencv_unity_mat = Matrix4x4.identity;
            opencv_unity_mat[1, 1] = -1;
            inv_view_mat = opencv_unity_mat * inv_view_mat * opencv_unity_mat;

            // The cameras are mounted sideways, and C0004 is used as reference coordinate system.
            Matrix4x4 dome_to_world = Matrix4x4.Rotate(Quaternion.Euler(domeToWorldRotation));
            inv_view_mat = dome_to_world * inv_view_mat;

            // Extract fov from camera_matrix
            Matrix4x4 camera_matrix = Utility.DecodeMatrix(camera_node["intrinsics"]["camera_matrix"]);
            int width = camera_node["intrinsics"]["resolution"][0].AsInt;
            int height = camera_node["intrinsics"]["resolution"][1].AsInt;
            float fov_x = Mathf.Rad2Deg * 2 * Mathf.Atan(0.5f * width / camera_matrix[0, 0]);
            float fov_y = Mathf.Rad2Deg * 2 * Mathf.Atan(0.5f * height / camera_matrix[1, 1]);

            if (uprightCameras)
            {
                (fov_y, fov_x) = (fov_x, fov_y);
                Vector3 local_up = inv_view_mat.inverse.MultiplyVector(this.transform.InverseTransformDirection(Vector3.up));
                if (local_up.x < 0)
                {
                    // world up = camera left
                    // rotate image 90° cw
                    // rotate camera 90° ccw
                    inv_view_mat *= Matrix4x4.Rotate(Quaternion.Euler(0, 0, 90));
                }
                else // local_up.x > 0
                {
                    // world up = camera right
                    // rotate image 90° ccw
                    // rotate camera 90° cw
                    inv_view_mat *= Matrix4x4.Rotate(Quaternion.Euler(0, 0, -90));
                }
            }

            // Set vertical field of view
            camera.fieldOfView = fov_y;

            gameObject.transform.SetLocalPositionAndRotation(inv_view_mat.GetPosition(), inv_view_mat.rotation);
        }
    }
}