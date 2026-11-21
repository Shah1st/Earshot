using System.Collections.Generic;
using System.Globalization;
using CityGenerator.Runtime;
using UnityEngine;
using UnityEngine.UI;

namespace Earshot.Simulation
{
    /// <summary>Tracks the taxi on City Generator's minimap and draws the planned road route.</summary>
    [DisallowMultipleComponent]
    public sealed class TaxiRouteNavigator : MonoBehaviour
    {
        [SerializeField] private TaxiRoutePlanner routePlanner;
        [SerializeField] private MinimapHUD minimap;
        [SerializeField] private Transform taxi;
        [SerializeField] private Color routeColor = new Color(0.22f, 0.84f, 1f);
        [SerializeField, Min(1f)] private float lineWidth = 5f;
        [SerializeField] private TaxiGpsReference gpsReference = new TaxiGpsReference();

        private TaxiRouteOverlay overlay;
        private TaxiRouteDriver driver;
        private Image gpsReadout;
        private Text gpsText;
        private float nextGpsRefresh;

        public TaxiGpsCoordinate ProjectGps(Vector3 worldPosition)
        {
            gpsReference ??= new TaxiGpsReference();
            return gpsReference.Project(worldPosition);
        }

        private void OnEnable()
        {
            if (routePlanner == null)
                routePlanner = GetComponent<TaxiRoutePlanner>();
            if (taxi == null)
                taxi = transform;
            driver = taxi.GetComponent<TaxiRouteDriver>();
            if (minimap == null)
                minimap = FindAnyObjectByType<MinimapHUD>();

            if (routePlanner == null || minimap == null)
            {
                Debug.LogWarning("Taxi navigator needs a TaxiRoutePlanner and a Minimap HUD.", this);
                return;
            }

            // MinimapHUD reads MarkerTarget in Start. Bind it during OnEnable so
            // the map has a tracked transform even in a scene without a PlayerController.
            minimap.MarkerTarget = taxi;

            if (overlay == null)
            {
                RawImage map = minimap.GetComponentInChildren<RawImage>(true);
                if (map == null)
                {
                    Debug.LogWarning("Taxi navigator could not find the map image inside the Minimap HUD.", minimap);
                    return;
                }

                var graphics = new GameObject("Taxi Route Overlay", typeof(RectTransform), typeof(CanvasRenderer));
                var rect = (RectTransform)graphics.transform;
                // Drawing below the map's sibling markers keeps them legible and
                // lets the existing minimap mask clip the highlighted route.
                rect.SetParent(map.rectTransform, false);
                rect.anchorMin = Vector2.zero;
                rect.anchorMax = Vector2.one;
                rect.pivot = new Vector2(0.5f, 0.5f);
                rect.offsetMin = Vector2.zero;
                rect.offsetMax = Vector2.zero;
                overlay = graphics.AddComponent<TaxiRouteOverlay>();
                overlay.raycastTarget = false;
                CreateGpsReadout((RectTransform)map.rectTransform.parent);
            }

            overlay.gameObject.SetActive(true);
            if (gpsReadout != null)
                gpsReadout.gameObject.SetActive(true);
            nextGpsRefresh = 0f;
        }

        private void LateUpdate()
        {
            if (overlay == null || minimap == null || taxi == null || routePlanner == null)
                return;

            overlay.SetRoute(routePlanner.CurrentRoute, driver, taxi.position, minimap.ViewRadiusMeters, routeColor, lineWidth);
            if (gpsText != null && Time.unscaledTime >= nextGpsRefresh)
            {
                TaxiGpsCoordinate fix = ProjectGps(taxi.position);
                gpsText.text = string.Format(CultureInfo.InvariantCulture,
                    "LAT {0:F5}    LON {1:F5}", fix.Latitude, fix.Longitude);
                nextGpsRefresh = Time.unscaledTime + 0.2f;
            }
        }

        private void CreateGpsReadout(RectTransform mapPanel)
        {
            var panel = new GameObject("GPS Readout", typeof(RectTransform), typeof(Image));
            var rect = (RectTransform)panel.transform;
            rect.SetParent(mapPanel, false);
            rect.anchorMin = Vector2.zero;
            rect.anchorMax = Vector2.right;
            rect.pivot = new Vector2(0.5f, 1f);
            rect.anchoredPosition = Vector2.down * 6f;
            rect.sizeDelta = new Vector2(0f, 26f);

            gpsReadout = panel.GetComponent<Image>();
            gpsReadout.color = new Color32(12, 23, 34, 235);
            gpsReadout.raycastTarget = false;
            // This strip sits under the circular map and should bypass its stencil mask.
            gpsReadout.maskable = false;

            var label = new GameObject("Coordinates", typeof(RectTransform), typeof(Text));
            gpsText = label.GetComponent<Text>();
            RectTransform labelRect = gpsText.rectTransform;
            labelRect.SetParent(rect, false);
            labelRect.anchorMin = Vector2.zero;
            labelRect.anchorMax = Vector2.one;
            labelRect.offsetMin = new Vector2(6f, 0f);
            labelRect.offsetMax = new Vector2(-6f, 0f);
            gpsText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            gpsText.fontSize = 13;
            gpsText.resizeTextForBestFit = true;
            gpsText.resizeTextMinSize = 9;
            gpsText.resizeTextMaxSize = 13;
            gpsText.alignment = TextAnchor.MiddleCenter;
            gpsText.color = new Color32(235, 247, 255, 255);
            gpsText.raycastTarget = false;
            gpsText.maskable = false;
        }

        private void OnDisable()
        {
            if (overlay != null)
                overlay.gameObject.SetActive(false);
            if (gpsReadout != null)
                gpsReadout.gameObject.SetActive(false);
        }

        private void OnDestroy()
        {
            if (overlay != null)
                Destroy(overlay.gameObject);
            if (gpsReadout != null)
                Destroy(gpsReadout.gameObject);
        }
    }

    [AddComponentMenu("")]
    internal sealed class TaxiRouteOverlay : MaskableGraphic
    {
        private static readonly Color32 RouteOutline = new Color32(10, 33, 51, 255);
        private static readonly Color32 StopColor = new Color32(238, 249, 255, 255);
        private static readonly Color32 DestinationColor = new Color32(255, 193, 75, 255);
        private static readonly Color32 TrailColor = new Color32(82, 187, 133, 255);
        private static readonly Color32 ChangeColor = new Color32(255, 134, 94, 255);

        private TaxiRoutePlanner.Route route;
        private IReadOnlyList<Vector3> trail;
        private int trailCount;
        private int rideSequence;
        private Vector3? changePosition;
        private Vector2 center;
        private float viewRadius;
        private float width;

        public void SetRoute(TaxiRoutePlanner.Route planned, TaxiRouteDriver driver, Vector3 taxiPosition,
            float radius, Color tint, float lineWidth)
        {
            Vector2 nextCenter = new Vector2(taxiPosition.x, taxiPosition.z);
            IReadOnlyList<Vector3> driven = driver != null ? driver.DrivenWaypoints : null;
            int sequence = driver != null ? driver.RideSequence : 0;
            Vector3? changedAt = driver != null ? driver.RouteChangePosition : null;
            int count = driven?.Count ?? 0;
            if (route == planned && center == nextCenter && viewRadius == radius && color == tint && width == lineWidth
                && rideSequence == sequence && trailCount == count && changePosition == changedAt)
                return;

            route = planned;
            trail = driven;
            trailCount = count;
            rideSequence = sequence;
            changePosition = changedAt;
            center = nextCenter;
            viewRadius = radius;
            width = lineWidth;
            color = tint;
            SetVerticesDirty();
        }

        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();
            if (route == null || viewRadius <= 0f)
                return;

            Rect rect = rectTransform.rect;
            if (rect.width <= 0f || rect.height <= 0f)
                return;

            // The package map stays north-up and shows ±ViewRadiusMeters on each axis.
            Vector2 scale = rect.size / (2f * viewRadius);
            Rect visibleBounds = Rect.MinMaxRect(rect.xMin - 8f, rect.yMin - 8f, rect.xMax + 8f, rect.yMax + 8f);
            float halfWidth = Mathf.Max(1f, width) * 0.5f;
            // The planned route and the traveled trail can overlap. Leave a visible dashed edge.
            DrawOriginalRoute(mesh, route.Waypoints, scale, visibleBounds, halfWidth + 2f);
            if (trailCount > 0)
                DrawDrivenTrail(mesh, scale, visibleBounds, halfWidth);

            IReadOnlyList<int> stops = route.StopWaypointIndices;
            for (int i = 1; i < stops.Count; i++)
            {
                Vector2 at = Project(route.Waypoints[stops[i]], scale);
                if (!visibleBounds.Contains(at))
                    continue;

                bool isDestination = i == stops.Count - 1;
                float radius = isDestination ? 6f : 4f;
                AddDisk(mesh, at, radius + 1.25f, RouteOutline);
                AddDisk(mesh, at, radius, isDestination ? DestinationColor : StopColor);
            }

            if (changePosition.HasValue)
            {
                Vector2 at = Project(changePosition.Value, scale);
                if (visibleBounds.Contains(at))
                {
                    AddDisk(mesh, at, 8f, RouteOutline);
                    AddDisk(mesh, at, 6.5f, ChangeColor);
                    AddDisk(mesh, at, 2.5f, RouteOutline);
                }
            }
        }

        private Vector2 Project(Vector3 point, Vector2 scale) =>
            new Vector2((point.x - center.x) * scale.x, (point.z - center.y) * scale.y);

        private void DrawOriginalRoute(VertexHelper mesh, IReadOnlyList<Vector3> points, Vector2 scale, Rect bounds, float radius)
        {
            const float dash = 9f;
            const float period = 15f;
            Vector2 from = Project(points[0], scale);
            float offset = 0f;
            for (int i = 1; i < points.Count; i++)
            {
                Vector2 to = Project(points[i], scale);
                float length = Vector2.Distance(from, to);
                if (length > 0.0001f && IsNearView(from, to, bounds))
                {
                    Vector2 direction = (to - from) / length;
                    for (float start = -offset; start < length; start += period)
                    {
                        float end = Mathf.Min(start + dash, length);
                        if (end > 0f)
                            DrawSegment(mesh, from + direction * Mathf.Max(0f, start), from + direction * end, radius, color);
                    }
                }
                offset = (offset + length) % period;
                from = to;
            }
        }

        private void DrawDrivenTrail(VertexHelper mesh, Vector2 scale, Rect bounds, float radius)
        {
            Vector2 previous = Project(trail[0], scale);
            for (int i = 1; i < trailCount; i++)
            {
                Vector2 next = Project(trail[i], scale);
                if (IsNearView(previous, next, bounds))
                    DrawSegment(mesh, previous, next, radius, TrailColor);

                if (bounds.Contains(next))
                    AddDisk(mesh, next, radius, TrailColor);
                previous = next;
            }
            DrawSegment(mesh, previous, Vector2.zero, radius, TrailColor);
        }

        private static bool IsNearView(Vector2 a, Vector2 b, Rect bounds) =>
            !(a.x < bounds.xMin && b.x < bounds.xMin || a.x > bounds.xMax && b.x > bounds.xMax
            || a.y < bounds.yMin && b.y < bounds.yMin || a.y > bounds.yMax && b.y > bounds.yMax);

        private static void DrawSegment(VertexHelper mesh, Vector2 a, Vector2 b, float radius, Color32 tint)
        {
            Vector2 direction = (b - a).normalized;
            if (direction == Vector2.zero)
                return;
            Vector2 normal = new Vector2(-direction.y, direction.x) * radius;
            int start = mesh.currentVertCount;
            mesh.AddVert(a - normal, tint, Vector2.zero);
            mesh.AddVert(a + normal, tint, Vector2.zero);
            mesh.AddVert(b + normal, tint, Vector2.zero);
            mesh.AddVert(b - normal, tint, Vector2.zero);
            mesh.AddTriangle(start, start + 1, start + 2);
            mesh.AddTriangle(start, start + 2, start + 3);
        }

        private static void AddDisk(VertexHelper mesh, Vector2 center, float radius, Color32 tint)
        {
            const int sides = 12;
            int start = mesh.currentVertCount;
            mesh.AddVert(center, tint, Vector2.zero);
            for (int i = 0; i < sides; i++)
            {
                float angle = -2f * Mathf.PI * i / sides;
                Vector2 offset = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle)) * radius;
                mesh.AddVert(center + offset, tint, Vector2.zero);
                mesh.AddTriangle(start, start + 1 + i, start + 1 + (i + 1) % sides);
            }
        }
    }
}
