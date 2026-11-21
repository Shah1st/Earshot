using System;
using System.Collections.Generic;
using CityGenerator.Runtime;
using UnityEngine;

namespace Earshot.Simulation
{
    /// <summary>
    /// Plans a shortest-distance taxi route on City Generator's directed lane graph.
    /// Does not move vehicles or change CarAgent's random-turn behaviour.
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class TaxiRoutePlanner : MonoBehaviour
    {
        [Header("City and endpoints")]
        [SerializeField] private TrafficNetwork network;
        [SerializeField] private Transform pickup;
        [SerializeField] private Transform destination;
        [SerializeField] private bool planOnStart = true;
        [Tooltip("Maximum horizontal distance from each marker to a road, in metres.")]
        [SerializeField, Min(0.1f)] private float maxSnapDistance = 40f;

        [Header("Route display (optional)")]
        [SerializeField] private LineRenderer routeLine;
        [SerializeField] private Color routeColor = Color.cyan;
        [SerializeField, Min(0.01f)] private float lineWidth = 0.4f;
        [SerializeField, Min(0f)] private float displayHeight = 0.25f;

        public Route CurrentRoute { get; private set; }
        public string LastError { get; private set; } = string.Empty;
        public event Action<Route> RouteChanged;

        public sealed class Route
        {
            /// <summary>World-space road positions, including the snapped endpoints.</summary>
            public IReadOnlyList<Vector3> Waypoints { get; }
            /// <summary>Ordered graph nodes between the snapped endpoints; empty for a direct same-lane route.</summary>
            public IReadOnlyList<int> NodeIds { get; }
            public float DistanceMeters { get; }
            public Vector3 RequestedPickup { get; }
            public Vector3 RequestedDestination { get; }
            /// <summary>Pickup, intermediate stops, destination, in the requested order.</summary>
            public IReadOnlyList<Vector3> RequestedStops { get; }
            /// <summary>Index in Waypoints of each snapped stop, including pickup and destination.</summary>
            public IReadOnlyList<int> StopWaypointIndices { get; }

            internal Route(List<Vector3> points, List<int> nodes, List<Vector3> stops, List<int> stopIndices)
            {
                Waypoints = points.AsReadOnly();
                NodeIds = nodes.AsReadOnly();
                RequestedStops = stops.AsReadOnly();
                StopWaypointIndices = stopIndices.AsReadOnly();
                RequestedPickup = stops[0];
                RequestedDestination = stops[stops.Count - 1];
                for (int i = 1; i < points.Count; i++)
                    DistanceMeters += Vector3.Distance(points[i - 1], points[i]);
            }
        }

        private struct RoadPoint
        {
            public int From;
            public int To;
            public float Fraction;
            public Vector3 Position;
        }

        private void Start()
        {
            if (planOnStart)
                PlanRoute();
        }

        [ContextMenu("Plan Taxi Route")]
        public void PlanRoute()
        {
            if (pickup == null || destination == null)
            {
                Fail("Assign Pickup and Destination transforms first.");
                return;
            }
            PlanRoute(pickup.position, destination.position);
        }

        /// <summary>Call again to replace the planned route after changing a destination.</summary>
        public bool PlanRoute(Vector3 start, Vector3 end)
        {
            return PlanRoute(new[] { start, end });
        }

        /// <summary>
        /// Visit every supplied point in order. The first is pickup and the last is destination.
        /// Each leg follows directed roads; the entire request fails if any stop is unreachable.
        /// </summary>
        public bool PlanRoute(IReadOnlyList<Vector3> orderedStops)
        {
            if (!TryBuildRoute(orderedStops, out Route planned, out string error))
                return Fail(error);

            SetPlannedRoute(planned);
            return true;
        }

        public void SetPlannedRoute(Route planned)
        {
            CurrentRoute = planned ?? throw new ArgumentNullException(nameof(planned));
            LastError = string.Empty;
            RefreshLine();
            Debug.Log($"Taxi route: {planned.DistanceMeters:F1} m, {planned.Waypoints.Count} waypoints, {planned.RequestedStops.Count} stops.", this);
            RouteChanged?.Invoke(planned);
        }

        /// <summary>Builds a route without changing the plan displayed by the navigator.</summary>
        public bool TryBuildRoute(IReadOnlyList<Vector3> orderedStops, out Route result, out string error)
        {
            result = null;
            error = string.Empty;
            if (orderedStops == null || orderedStops.Count < 2)
            {
                error = "Supply at least two stops: pickup and destination.";
                return false;
            }

            var stops = new List<Vector3>(orderedStops.Count);
            for (int i = 0; i < orderedStops.Count; i++)
            {
                if (!IsFinite(orderedStops[i]))
                {
                    error = $"Stop {i + 1} must contain finite coordinates.";
                    return false;
                }
                stops.Add(orderedStops[i]);
            }

            if (network == null)
                network = FindAnyObjectByType<TrafficNetwork>();
            if (network == null)
            {
                error = "No TrafficNetwork found. Generate a city with a traffic network first.";
                return false;
            }

            // Snap each stop once, so arrival and departure share exactly the same lane.
            var roadStops = new RoadPoint[stops.Count];
            for (int i = 0; i < stops.Count; i++)
            {
                if (!TrySnap(stops[i], out roadStops[i]))
                {
                    error = $"Stop {i + 1} is too far from a road, or the network has no road segments.";
                    return false;
                }
            }

            var nodeIds = new List<int>();
            var points = new List<Vector3> { roadStops[0].Position };
            var stopIndices = new List<int> { 0 };

            for (int leg = 0; leg < roadStops.Length - 1; leg++)
            {
                RoadPoint first = roadStops[leg];
                RoadPoint last = roadStops[leg + 1];
                // Forward travel within one lane needs no crossing. Backwards travel
                // must find a legal loop rather than drawing a wrong-way shortcut.
                bool direct = first.From == last.From && first.To == last.To
                    && last.Fraction >= first.Fraction;
                if (!direct)
                {
                    var legNodes = new List<int>();
                    if (!TryFindNodePath(first.To, last.From, legNodes))
                    {
                        error = $"No directed road route from stop {leg + 1} to stop {leg + 2}. Try another lane or street.";
                        return false;
                    }
                    foreach (int node in legNodes)
                    {
                        if (nodeIds.Count == 0 || nodeIds[nodeIds.Count - 1] != node)
                            nodeIds.Add(node);
                        AddPoint(points, network.GetNode(node).Position);
                    }
                }
                AddPoint(points, last.Position);
                stopIndices.Add(points.Count - 1);
            }

            result = new Route(points, nodeIds, stops, stopIndices);
            return true;
        }

        // Snap only to real street segments, not arbitrary nodes in holes of a custom city.
        // Markers are horizontally projected onto a lane; no off-road line is added to the route.
        private bool TrySnap(Vector3 requested, out RoadPoint result)
        {
            result = default;
            float bestSquared = Mathf.Max(0.1f, maxSnapDistance);
            bestSquared *= bestSquared;
            bool found = false;
            for (int i = 0; i < network.LaneSegmentCount; i++)
            {
                network.GetLaneSegment(i, out int from, out int to);
                Vector3 a = network.GetNode(from).Position;
                Vector3 b = network.GetNode(to).Position;
                Vector3 direction = b - a;
                direction.y = 0f;
                float lengthSquared = direction.sqrMagnitude;
                if (lengthSquared < 0.000001f)
                    continue;
                Vector3 offset = requested - a;
                offset.y = 0f;
                float fraction = Mathf.Clamp01(Vector3.Dot(offset, direction) / lengthSquared);
                Vector3 projected = Vector3.Lerp(a, b, fraction);
                Vector3 difference = requested - projected;
                difference.y = 0f;
                if (difference.sqrMagnitude > bestSquared)
                    continue;
                // Stable tie-breaking: retain the first equally near lane.
                if (found && difference.sqrMagnitude >= bestSquared)
                    continue;
                bestSquared = difference.sqrMagnitude;
                result = new RoadPoint { From = from, To = to, Fraction = fraction, Position = projected };
                found = true;
            }
            return found;
        }

        // A* with Euclidean distance as an admissible heuristic. Exit.Weight is a random
        // traffic preference, NOT a distance, so it must not be used as an edge cost.
        // Linear open-set selection keeps this dependency-free; search runs only on request.
        private bool TryFindNodePath(int start, int goal, List<int> path)
        {
            int count = network.NodeCount;
            var cost = new float[count];
            var parent = new int[count];
            var closed = new bool[count];
            var open = new List<int> { start };
            var inOpen = new bool[count];
            for (int i = 0; i < count; i++)
            {
                cost[i] = float.PositiveInfinity;
                parent[i] = -1;
            }
            cost[start] = 0f;
            inOpen[start] = true;
            Vector3 goalPosition = network.GetNode(goal).Position;

            while (open.Count > 0)
            {
                int best = 0;
                float bestScore = float.PositiveInfinity;
                for (int i = 0; i < open.Count; i++)
                {
                    int candidate = open[i];
                    float score = cost[candidate]
                        + Vector3.Distance(network.GetNode(candidate).Position, goalPosition);
                    if (score < bestScore)
                    {
                        bestScore = score;
                        best = i;
                    }
                }
                int current = open[best];
                open.RemoveAt(best);
                inOpen[current] = false;
                if (current == goal)
                {
                    for (int node = goal; node >= 0; node = parent[node])
                        path.Add(node);
                    path.Reverse();
                    return true;
                }
                closed[current] = true;
                TrafficNetwork.Node currentNode = network.GetNode(current);
                foreach (TrafficNetwork.Exit exit in currentNode.Exits)
                {
                    int next = exit.Node;
                    if (next < 0 || next >= count || closed[next])
                        continue;
                    float nextCost = cost[current]
                        + Vector3.Distance(currentNode.Position, network.GetNode(next).Position);
                    if (nextCost >= cost[next])
                        continue;
                    cost[next] = nextCost;
                    parent[next] = current;
                    if (!inOpen[next])
                    {
                        inOpen[next] = true;
                        open.Add(next);
                    }
                }
            }
            return false;
        }

        private static bool IsFinite(Vector3 value) =>
            !float.IsNaN(value.x) && !float.IsInfinity(value.x)
            && !float.IsNaN(value.y) && !float.IsInfinity(value.y)
            && !float.IsNaN(value.z) && !float.IsInfinity(value.z);

        private static void AddPoint(List<Vector3> points, Vector3 point)
        {
            if ((points[points.Count - 1] - point).sqrMagnitude > 0.000001f)
                points.Add(point);
        }

        private bool Fail(string message)
        {
            CurrentRoute = null;
            LastError = message;
            RefreshLine();
            Debug.LogWarning("Taxi route: " + message, this);
            RouteChanged?.Invoke(null);
            return false;
        }

        private void RefreshLine()
        {
            if (routeLine == null)
                return;
            routeLine.useWorldSpace = true;
            routeLine.loop = false;
            routeLine.startWidth = routeLine.endWidth = lineWidth;
            routeLine.startColor = routeLine.endColor = routeColor;
            routeLine.positionCount = CurrentRoute == null ? 0 : CurrentRoute.Waypoints.Count;
            for (int i = 0; i < routeLine.positionCount; i++)
                routeLine.SetPosition(i, CurrentRoute.Waypoints[i] + Vector3.up * displayHeight);
        }

        private void OnDrawGizmos()
        {
            if (CurrentRoute == null)
                return;
            Gizmos.color = routeColor;
            var points = CurrentRoute.Waypoints;
            for (int i = 1; i < points.Count; i++)
                Gizmos.DrawLine(points[i - 1] + Vector3.up * displayHeight,
                    points[i] + Vector3.up * displayHeight);
        }
    }
}
