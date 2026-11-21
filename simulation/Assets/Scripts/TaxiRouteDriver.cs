using System;
using System.Collections.Generic;
using CityGenerator.Runtime;
using UnityEngine;

namespace Earshot.Simulation
{
    /// <summary>
    /// Scripted simulation driver: teleports to the snapped pickup, follows every route
    /// waypoint, and stops at the destination. Does not implement traffic-light or obstacle AI.
    /// Attach to the vehicle root, on the same object as its CarAgent/Rigidbody, if present.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class TaxiRouteDriver : MonoBehaviour
    {
        [Header("Route")]
        [SerializeField] private TaxiRoutePlanner routePlanner;
        [SerializeField] private bool driveOnStart = true;

        [Header("Driving")]
        [SerializeField, Min(0.1f)] private float speedKmh = 36f;
        [SerializeField, Min(0.1f)] private float acceleration = 3f;
        [SerializeField, Min(0.1f)] private float braking = 5f;
        [SerializeField, Min(1f)] private float turnSpeed = 180f;
        [Tooltip("Added to the route height. Adjust if the vehicle's pivot is above/below its wheels.")]
        [SerializeField] private float heightOffset;

        public bool IsDriving { get; private set; }
        public bool HasArrived { get; private set; }
        public float SpeedMetersPerSecond { get; private set; }
        public float SpeedKmh => SpeedMetersPerSecond * 3.6f;
        public int NextWaypointIndex { get; private set; }
        public TaxiRoutePlanner.Route ActiveRoute => route;
        public IReadOnlyList<Vector3> DrivenWaypoints => drivenWaypoints;
        public Vector3? RouteChangePosition { get; private set; }
        public int RideSequence { get; private set; }
        public bool IsHolding { get; private set; }
        public event Action Arrived;
        public event Action<int> WaypointReached;

        private TaxiRoutePlanner subscribedPlanner;
        private TaxiRoutePlanner.Route route;
        private Rigidbody body;
        private CarAgent carAgent;
        private bool started;
        private bool followRouteChanges;
        private bool ownsVehicle;
        private bool previousCarEnabled;
        private bool previousKinematic;
        private bool previousGravity;
        private RigidbodyInterpolation previousInterpolation;
        private readonly List<Vector3> drivenWaypoints = new List<Vector3>();
        private TaxiRoutePlanner.Route pendingDetour;
        private int detourWaypointIndex;

        private void OnEnable()
        {
            BindPlanner();
            if (started && driveOnStart)
                StartRide();
        }

        private void Start()
        {
            started = true;
            if (driveOnStart)
                StartRide();
        }

        private bool BindPlanner()
        {
            if (routePlanner == null)
                routePlanner = FindAnyObjectByType<TaxiRoutePlanner>();
            if (subscribedPlanner != routePlanner)
            {
                if (subscribedPlanner != null)
                    subscribedPlanner.RouteChanged -= OnRouteChanged;
                subscribedPlanner = routePlanner;
                if (subscribedPlanner != null)
                    subscribedPlanner.RouteChanged += OnRouteChanged;
            }
            return routePlanner != null;
        }

        [ContextMenu("Start / Restart Taxi Ride")]
        public void StartRide()
        {
            if (!Application.isPlaying || !isActiveAndEnabled)
                return;
            if (!BindPlanner())
            {
                Debug.LogWarning("Taxi driver: assign a TaxiRoutePlanner first.", this);
                return;
            }
            followRouteChanges = true;
            // Handles either Unity Start order without calculating the same route twice.
            if (routePlanner.CurrentRoute != null)
                BeginRoute(routePlanner.CurrentRoute);
            else
                routePlanner.PlanRoute(); // RouteChanged starts the ride if planning succeeds.
        }

        private void OnRouteChanged(TaxiRoutePlanner.Route newRoute)
        {
            if (started && followRouteChanges)
                BeginRoute(newRoute);
        }

        public void StartScenarioRoute(TaxiRoutePlanner.Route plannedRoute)
        {
            followRouteChanges = false;
            BeginRoute(plannedRoute);
        }

        public void BrakeToStop() => IsHolding = true;

        public void ResumeDriving() => IsHolding = false;

        public bool QueueDetour(TaxiRoutePlanner.Route detour, int atWaypoint)
        {
            if (!IsDriving || pendingDetour != null || RouteChangePosition.HasValue || detour == null
                || atWaypoint < NextWaypointIndex || atWaypoint >= route.Waypoints.Count - 1
                || atWaypoint >= detour.Waypoints.Count - 1
                || (route.Waypoints[atWaypoint] - detour.Waypoints[atWaypoint]).sqrMagnitude > 0.000001f)
                return false;

            pendingDetour = detour;
            detourWaypointIndex = atWaypoint;
            return true;
        }

        private void BeginRoute(TaxiRoutePlanner.Route newRoute)
        {
            IsDriving = false;
            HasArrived = false;
            IsHolding = false;
            SpeedMetersPerSecond = 0f;
            route = newRoute;
            NextWaypointIndex = 0;
            RideSequence++;
            pendingDetour = null;
            RouteChangePosition = null;
            drivenWaypoints.Clear();
            if (route == null || route.Waypoints.Count == 0)
                return;

            TakeVehicleControl();
            Vector3 start = PositionAt(0);
            drivenWaypoints.Add(route.Waypoints[0]);
            Quaternion rotation = transform.rotation;
            for (int i = 1; i < route.Waypoints.Count; i++)
            {
                Vector3 direction = PositionAt(i) - start;
                direction.y = 0f;
                if (direction.sqrMagnitude < 0.000001f)
                    continue;
                rotation = Quaternion.LookRotation(direction, Vector3.up);
                break;
            }
            // Reset the physics pose directly so interpolation starts at the pickup.
            if (body != null)
            {
                body.position = start;
                body.rotation = rotation;
            }
            else
                transform.SetPositionAndRotation(start, rotation);
            NextWaypointIndex = 1;
            IsDriving = true;
            WaypointReached?.Invoke(0);
            if (NextWaypointIndex >= route.Waypoints.Count)
                FinishRide();
        }

        private void TakeVehicleControl()
        {
            if (ownsVehicle)
                return;
            carAgent = GetComponent<CarAgent>();
            body = GetComponent<Rigidbody>();
            if (carAgent != null)
            {
                previousCarEnabled = carAgent.enabled;
                carAgent.enabled = false; // Also unregisters it from TrafficManager.
            }
            if (body != null)
            {
                previousKinematic = body.isKinematic;
                previousGravity = body.useGravity;
                previousInterpolation = body.interpolation;
                if (!body.isKinematic)
                {
                    body.linearVelocity = Vector3.zero;
                    body.angularVelocity = Vector3.zero;
                }
                body.useGravity = false;
                body.isKinematic = true;
                body.interpolation = RigidbodyInterpolation.Interpolate;
            }
            ownsVehicle = true;
        }

        private Vector3 PositionAt(int index) =>
            route.Waypoints[index] + Vector3.up * heightOffset;

        private void Update()
        {
            // The demo taxi is a transform-driven model. Move it every frame so
            // the LateUpdate follow camera always sees its current pose.
            if (body == null)
                AdvanceRide(Time.deltaTime);
        }

        private void FixedUpdate()
        {
            if (body != null)
                AdvanceRide(Time.fixedDeltaTime);
        }

        private void AdvanceRide(float deltaTime)
        {
            if (!IsDriving || route == null || deltaTime <= 0f)
                return;
            if (IsHolding && SpeedMetersPerSecond <= 0f)
                return;

            Vector3 position = body != null ? body.position : transform.position;
            float remaining = Vector3.Distance(position, PositionAt(NextWaypointIndex));
            for (int i = NextWaypointIndex + 1; i < route.Waypoints.Count; i++)
                remaining += Vector3.Distance(PositionAt(i - 1), PositionAt(i));

            float desiredSpeed = IsHolding ? 0f : Mathf.Min(Mathf.Max(0.1f, speedKmh) / 3.6f,
                Mathf.Sqrt(2f * Mathf.Max(0.1f, braking) * remaining));
            float oldSpeed = SpeedMetersPerSecond;
            float rate = desiredSpeed < oldSpeed ? braking : acceleration;
            SpeedMetersPerSecond = Mathf.MoveTowards(oldSpeed, desiredSpeed,
                Mathf.Max(0.1f, rate) * deltaTime);
            float travel = (oldSpeed + SpeedMetersPerSecond) * 0.5f * deltaTime;
            Vector3 facing = Vector3.zero;

            // Consume the distance budget across short/coincident waypoints without
            // skipping stops, overshooting a corner, or varying speed with waypoint density.
            while (NextWaypointIndex < route.Waypoints.Count)
            {
                Vector3 delta = PositionAt(NextWaypointIndex) - position;
                float distance = delta.magnitude;
                if (distance > 0.0001f)
                    facing = delta;
                if (distance <= travel || distance <= 0.0001f)
                {
                    position = PositionAt(NextWaypointIndex);
                    travel = Mathf.Max(0f, travel - distance);
                    int reached = NextWaypointIndex++;
                    drivenWaypoints.Add(route.Waypoints[reached]);
                    if (pendingDetour != null && reached == detourWaypointIndex)
                    {
                        RouteChangePosition = route.Waypoints[reached];
                        route = pendingDetour;
                        pendingDetour = null;
                    }
                    WaypointReached?.Invoke(reached);
                }
                else
                {
                    position += delta / distance * travel;
                    break;
                }
            }

            facing.y = 0f;
            Quaternion rotation = body != null ? body.rotation : transform.rotation;
            if (facing.sqrMagnitude > 0.000001f)
                rotation = Quaternion.RotateTowards(rotation,
                    Quaternion.LookRotation(facing, Vector3.up),
                    Mathf.Max(1f, turnSpeed) * deltaTime);
            if (body != null)
                body.Move(position, rotation);
            else
                transform.SetPositionAndRotation(position, rotation);

            if (NextWaypointIndex >= route.Waypoints.Count)
                FinishRide();
        }

        private void FinishRide()
        {
            IsDriving = false;
            IsHolding = false;
            HasArrived = true;
            SpeedMetersPerSecond = 0f;
            Debug.Log("Taxi driver: destination reached.", this);
            Arrived?.Invoke();
        }

        [ContextMenu("Stop Taxi Ride")]
        public void StopRide()
        {
            followRouteChanges = false;
            IsDriving = false;
            IsHolding = false;
            pendingDetour = null;
            SpeedMetersPerSecond = 0f;
        }

        private void OnDisable()
        {
            StopRide();
            if (subscribedPlanner != null)
                subscribedPlanner.RouteChanged -= OnRouteChanged;
            subscribedPlanner = null;
            if (!ownsVehicle)
                return;
            if (body != null)
            {
                body.interpolation = previousInterpolation;
                body.isKinematic = previousKinematic;
                body.useGravity = previousGravity;
            }
            if (carAgent != null)
                carAgent.enabled = previousCarEnabled;
            ownsVehicle = false;
        }
    }
}
