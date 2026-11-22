using UnityEngine;

namespace Earshot.Simulation
{
    [DisallowMultipleComponent]
    [RequireComponent(typeof(Camera))]
    public sealed class TaxiFollowCamera : MonoBehaviour
    {
        [SerializeField] private Transform target;
        [SerializeField, Min(0.1f)] private float height = 9f;
        [SerializeField, Min(0.1f)] private float distance = 10f;
        [SerializeField, Min(0f)] private float lookAhead = 4f;
        [SerializeField, Min(0f)] private float followSmoothTime = 0.16f;

        private const float SnapDistance = 25f;
        private Vector3 followPosition;
        private Vector3 followVelocity;
        private float followYaw;
        private float yawVelocity;
        private Transform lastTarget;

        private void OnEnable()
        {
            lastTarget = null;
            followVelocity = Vector3.zero;
            yawVelocity = 0f;
        }

        private void LateUpdate()
        {
            if (target == null)
                return;

            Vector3 targetPosition = target.position;
            Vector3 targetForward = target.forward;
            float targetYaw = Mathf.Atan2(targetForward.x, targetForward.z) * Mathf.Rad2Deg;

            // The first frame and pickup teleports should jump straight into the chase view.
            if (lastTarget != target || (targetPosition - followPosition).sqrMagnitude > SnapDistance * SnapDistance
                || followSmoothTime <= 0f)
            {
                followPosition = targetPosition;
                followVelocity = Vector3.zero;
                followYaw = targetYaw;
                yawVelocity = 0f;
                lastTarget = target;
            }
            else if (Time.deltaTime > 0f)
            {
                followPosition = Vector3.SmoothDamp(followPosition, targetPosition, ref followVelocity, followSmoothTime);
                followYaw = Mathf.SmoothDampAngle(followYaw, targetYaw, ref yawVelocity, followSmoothTime);
            }

            Vector3 forward = Quaternion.Euler(0f, followYaw, 0f) * Vector3.forward;
            Vector3 cameraPosition = followPosition + Vector3.up * height - forward * distance;
            Vector3 aim = targetPosition + Vector3.up + forward * lookAhead;
            transform.SetPositionAndRotation(cameraPosition, Quaternion.LookRotation(aim - cameraPosition, Vector3.up));
        }
    }
}
