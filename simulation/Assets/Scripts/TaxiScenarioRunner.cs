using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using UnityEngine;

namespace Earshot.Simulation
{
    [DefaultExecutionOrder(-50)]
    [DisallowMultipleComponent]
    [RequireComponent(typeof(TaxiRoutePlanner), typeof(TaxiRouteDriver))]
    public sealed class TaxiScenarioRunner : MonoBehaviour
    {
        [SerializeField] private TextAsset scenarioConfig;
        [SerializeField, Min(0)] private int scenarioIndex;
        [SerializeField] private bool streamToAndroid;

        [Serializable] private sealed class Config { public Marker[] markers; public Scenario[] scenarios; }
        [Serializable] private sealed class Marker { public string id; public Vector3 position; }
        [Serializable] private sealed class Scenario
        {
            public string id;
            public string[] plannedRoute;
            public string[] drivingRoute;
            public RideEvent[] events;
        }
        [Serializable] private sealed class RideEvent
        {
            public string atMarker;
            public string playAudio;
            public string playReferenceRide;
            public Cue[] cues;
        }
        [Serializable] private sealed class Cue { public float atSeconds; public string action; }

        private sealed class PreparedEvent
        {
            public RideEvent Event;
            public int PlannedIndex;
            public int AlternateIndex;
            public AudioClip Clip;
            public byte[] ReferencePcm;
            public bool Fired;
        }

        private struct ScheduledCue { public double Time; public string Action; }

        private TaxiRoutePlanner planner;
        private TaxiRouteDriver driver;
        private TaxiRouteNavigator navigator;
        private TaxiRoutePlanner.Route alternate;
        private int forkIndex;
        private List<PreparedEvent> events = new List<PreparedEvent>();
        private readonly List<ScheduledCue> scheduled = new List<ScheduledCue>();
        private readonly List<AudioSource> audioSources = new List<AudioSource>();
        private readonly List<AudioClip> generatedClips = new List<AudioClip>();
        private const double AudioScheduleLead = 0.08;
        private TaxiReplayServer server;
        private ReferenceRide streamRide;
        private TaxiRoutePlanner.Route readyRoute;
        private float readyApproachSpeed;
        private double audioStart;
        private string streamError;
        private Config config;
        private readonly TaxiScenarioMenu menu = new TaxiScenarioMenu();
        private string[] scenarioIds = Array.Empty<string>();
        private int selectedScenario = -1;

        private void Awake()
        {
            planner = GetComponent<TaxiRoutePlanner>();
            driver = GetComponent<TaxiRouteDriver>();
            navigator = GetComponent<TaxiRouteNavigator>();
        }

        private void OnEnable() => driver.WaypointReached += OnWaypointReached;

        private void Start()
        {
            try
            {
                config = scenarioConfig != null ? JsonUtility.FromJson<Config>(scenarioConfig.text) : null;
                if (config?.scenarios == null || config.scenarios.Length == 0 || config.markers == null)
                    throw new ArgumentException("Assign a JSON config with markers and scenarios.");
                scenarioIds = config.scenarios.Select(scenario => scenario?.id ?? "").ToArray();
            }
            catch (ArgumentException error)
            {
                streamError = error.Message;
            }
        }

        private void OnDisable()
        {
            driver.WaypointReached -= OnWaypointReached;
            ClearTimeline();
        }

        [ContextMenu("Start / Restart Selected Scenario")]
        public void StartSelectedScenario() => LoadScenario(scenarioIndex);

        [ContextMenu("Next Scenario")]
        public void NextScenario()
        {
            if (scenarioIds.Length != 0)
                LoadScenario((Mathf.Clamp(scenarioIndex, 0, scenarioIds.Length - 1) + 1) % scenarioIds.Length);
        }

        private void LoadScenario(int index)
        {
            if (!Application.isPlaying || !isActiveAndEnabled || scenarioIds.Length == 0)
                return;
            try
            {
                if (index < 0 || index >= config.scenarios.Length || config.scenarios[index] == null)
                    throw new ArgumentException($"Scenario index {index} is missing.");

                var markers = new Dictionary<string, Vector3>(StringComparer.Ordinal);
                foreach (Marker marker in config.markers)
                {
                    if (marker == null || string.IsNullOrWhiteSpace(marker.id) || markers.ContainsKey(marker.id))
                        throw new ArgumentException("Every marker must have a unique, nonempty id.");
                    markers.Add(marker.id, marker.position);
                }

                Scenario scenario = config.scenarios[index];
                TaxiRoutePlanner.Route planned = BuildRoute(scenario.plannedRoute, markers, "plannedRoute");
                TaxiRoutePlanner.Route detour = null;
                int fork = -1;
                if (scenario.drivingRoute != null && scenario.drivingRoute.Length > 0)
                {
                    detour = BuildRoute(scenario.drivingRoute, markers, "drivingRoute");
                    if (scenario.plannedRoute[0] != scenario.drivingRoute[0]
                        || scenario.plannedRoute[scenario.plannedRoute.Length - 1] != scenario.drivingRoute[scenario.drivingRoute.Length - 1])
                        throw new ArgumentException("A detour must keep the same pickup and destination.");
                    fork = FindFork(planned, detour);
                    if (fork < 1)
                        throw new ArgumentException("The two routes must share an approach with at least one waypoint after pickup, then diverge.");
                }

                var prepared = PrepareEvents(scenario, planned, detour);
                float approachSpeed = GetApproachSpeedLimit(prepared, planned, fork);
                ClearTimeline();
                streamError = null;
                scenarioIndex = index;
                alternate = detour;
                forkIndex = fork;
                events = prepared;
                if (streamToAndroid)
                {
                    driver.StopRide();
                    streamRide = PrepareStream(scenario.id, planned, prepared);
                    server = new TaxiReplayServer(5005);
                    readyRoute = planned;
                    readyApproachSpeed = approachSpeed;
                    Debug.Log($"Taxi city scene waiting for Android on 127.0.0.1:{server.Port}.", this);
                }
                else
                {
                    driver.StartScenarioRoute(planned, approachSpeed);
                }
                planner.SetPlannedRoute(planned);
                selectedScenario = index;
                Debug.Log($"Taxi scenario {index + 1}/{config.scenarios.Length}: {scenario.id}", this);
            }
            catch (Exception error) when (error is ArgumentException || error is IOException
                || error is InvalidDataException || error is SocketException)
            {
                streamError = error.Message;
                Debug.LogError($"Taxi scenario: {error.Message}", this);
            }
        }

        private ReferenceRide PrepareStream(string id, TaxiRoutePlanner.Route planned, List<PreparedEvent> prepared)
        {
            if (navigator == null)
                throw new ArgumentException("Assign a TaxiRouteNavigator for Android GPS streaming.");
            var audio = prepared.Where(entry => entry.ReferencePcm != null || entry.Clip != null).ToArray();
            if (audio.Length != 1 || audio[0].PlannedIndex != 0 || audio[0].ReferencePcm == null)
                throw new ArgumentException("Android streaming needs one reference WAV that starts at pickup.");

            var route = planned.Waypoints.Select(position =>
            {
                TaxiGpsCoordinate point = navigator.ProjectGps(position);
                return new ReplayCoordinate { lat = point.Latitude, lon = point.Longitude };
            }).ToArray();
            return new ReferenceRide(id, route, audio[0].ReferencePcm);
        }

        private ReplayGpsFix SampleGps(int tick)
        {
            TaxiGpsCoordinate point = navigator.ProjectGps(driver.WorldPosition);
            return new ReplayGpsFix
            {
                t = tick * ReferenceRide.TickSeconds,
                lat = point.Latitude,
                lon = point.Longitude,
                speed_kmh = driver.SpeedKmh
            };
        }

        private TaxiRoutePlanner.Route BuildRoute(string[] ids, Dictionary<string, Vector3> markers, string field)
        {
            if (ids == null || ids.Length < 2)
                throw new ArgumentException($"{field} needs at least two marker ids.");
            var positions = new Vector3[ids.Length];
            for (int i = 0; i < ids.Length; i++)
            {
                if (string.IsNullOrEmpty(ids[i]) || !markers.TryGetValue(ids[i], out positions[i]))
                    throw new ArgumentException($"Unknown marker '{ids[i]}' in {field}.");
            }
            if (!planner.TryBuildRoute(positions, out TaxiRoutePlanner.Route route, out string error))
                throw new ArgumentException($"{field}: {error}");
            return route;
        }

        private static int FindFork(TaxiRoutePlanner.Route planned, TaxiRoutePlanner.Route detour)
        {
            int count = Math.Min(planned.Waypoints.Count, detour.Waypoints.Count);
            for (int i = 0; i < count; i++)
                if ((planned.Waypoints[i] - detour.Waypoints[i]).sqrMagnitude > 0.000001f)
                    return i - 1;
            return -1;
        }

        private static List<PreparedEvent> PrepareEvents(Scenario scenario, TaxiRoutePlanner.Route planned, TaxiRoutePlanner.Route detour)
        {
            var prepared = new List<PreparedEvent>();
            bool hasSwitch = false;
            foreach (RideEvent rideEvent in scenario.events ?? Array.Empty<RideEvent>())
            {
                if (rideEvent == null || string.IsNullOrWhiteSpace(rideEvent.atMarker))
                    throw new ArgumentException("Every event needs an atMarker id from one of its routes.");
                int original = Array.IndexOf(scenario.plannedRoute, rideEvent.atMarker);
                int alternate = detour == null ? -1 : Array.IndexOf(scenario.drivingRoute, rideEvent.atMarker);
                if (original < 0 && alternate < 0)
                    throw new ArgumentException($"Event marker '{rideEvent.atMarker}' is missing from both routes.");
                byte[] referencePcm = null;
                if (!string.IsNullOrEmpty(rideEvent.playReferenceRide))
                {
                    if (!string.IsNullOrEmpty(rideEvent.playAudio))
                        throw new ArgumentException("Choose one audio source per event.");
                    string path = Path.Combine(Application.streamingAssetsPath, "ReferenceRides", rideEvent.playReferenceRide + ".wav");
                    referencePcm = ReferenceRide.ReadPcmWave(path);
                }
                foreach (Cue cue in rideEvent.cues ?? Array.Empty<Cue>())
                {
                    if (cue == null || cue.atSeconds < 0f || float.IsNaN(cue.atSeconds) || float.IsInfinity(cue.atSeconds))
                        throw new ArgumentException("Event cue times must be finite and nonnegative.");
                    switch (cue.action)
                    {
                        case "brake_to_stop":
                        case "resume":
                            break;
                        case "switch_route":
                            if (detour == null || hasSwitch)
                                throw new ArgumentException("Use exactly one switch_route cue when providing a drivingRoute.");
                            hasSwitch = true;
                            break;
                        default:
                            throw new ArgumentException($"Unknown cue action '{cue.action}'.");
                    }
                }
                prepared.Add(new PreparedEvent
                {
                    Event = rideEvent,
                    PlannedIndex = original < 0 ? -1 : planned.StopWaypointIndices[original],
                    AlternateIndex = alternate < 0 ? -1 : detour.StopWaypointIndices[alternate],
                    Clip = string.IsNullOrEmpty(rideEvent.playAudio) ? null : Resources.Load<AudioClip>("TaxiAudio/" + rideEvent.playAudio),
                    ReferencePcm = referencePcm
                });
            }
            if (detour != null && !hasSwitch)
                throw new ArgumentException("A drivingRoute requires a switch_route cue.");
            return prepared;
        }

        private static float GetApproachSpeedLimit(List<PreparedEvent> prepared, TaxiRoutePlanner.Route planned, int fork)
        {
            if (fork < 1)
                return float.PositiveInfinity;

            foreach (PreparedEvent entry in prepared)
            {
                if (entry.PlannedIndex != 0 || entry.ReferencePcm == null || entry.Event.cues == null)
                    continue;
                Cue turn = Array.Find(entry.Event.cues, cue => cue.action == "switch_route");
                if (turn == null)
                    continue;

                float distanceToFork = 0f;
                for (int i = 1; i <= fork; i++)
                    distanceToFork += Vector3.Distance(planned.Waypoints[i - 1], planned.Waypoints[i]);
                // Leave time to process the audio cue before the taxi reaches the last common waypoint.
                float earliestForkTime = turn.atSeconds + (float)AudioScheduleLead + 0.25f;
                return distanceToFork / earliestForkTime * 3.6f;
            }
            return float.PositiveInfinity;
        }

        private void OnWaypointReached(int index)
        {
            bool tookDetour = alternate != null && driver.ActiveRoute == alternate;
            foreach (PreparedEvent entry in events)
            {
                if (entry.Fired || index != (tookDetour ? entry.AlternateIndex : entry.PlannedIndex))
                    continue;
                entry.Fired = true;
                double startTime = AudioSettings.dspTime;
                AudioClip clip = entry.Clip;
                if (entry.ReferencePcm != null)
                {
                    float[] samples = ReferenceRide.PlaybackSamples(entry.ReferencePcm);
                    clip = AudioClip.Create(entry.Event.playReferenceRide, samples.Length, 1, ReferenceRide.SampleRate, false);
                    clip.SetData(samples, 0);
                    generatedClips.Add(clip);
                }
                if (clip != null)
                {
                    startTime = AudioSettings.dspTime + AudioScheduleLead;
                    if (server != null && entry.ReferencePcm != null)
                        audioStart = startTime;
                    var source = gameObject.AddComponent<AudioSource>();
                    audioSources.Add(source);
                    source.playOnAwake = false;
                    source.clip = clip;
                    source.PlayScheduled(startTime);
                }
                else if (!string.IsNullOrEmpty(entry.Event.playAudio))
                    Debug.LogWarning($"Taxi scenario audio missing: Resources/TaxiAudio/{entry.Event.playAudio}. Timed cues will still run.", this);

                foreach (Cue cue in entry.Event.cues ?? Array.Empty<Cue>())
                    scheduled.Add(new ScheduledCue { Time = startTime + cue.atSeconds, Action = cue.action });
            }
            // Keep insertion order when actions share a timestamp.
            for (int i = 1; i < scheduled.Count; i++)
            {
                ScheduledCue cue = scheduled[i];
                int j = i - 1;
                while (j >= 0 && scheduled[j].Time > cue.Time)
                {
                    scheduled[j + 1] = scheduled[j];
                    j--;
                }
                scheduled[j + 1] = cue;
            }
        }

        private void Update()
        {
            if (readyRoute != null && server?.Status == TaxiReplayServer.State.Connected)
            {
                driver.StartScenarioRoute(readyRoute, readyApproachSpeed);
                readyRoute = null;
                server.Begin(streamRide, SampleGps);
            }

            double now = AudioSettings.dspTime;
            while (scheduled.Count > 0 && scheduled[0].Time <= now)
            {
                string action = scheduled[0].Action;
                scheduled.RemoveAt(0);
                switch (action)
                {
                    case "brake_to_stop": driver.BrakeToStop(); break;
                    case "resume": driver.ResumeDriving(); break;
                    case "switch_route":
                        if (!driver.QueueDetour(alternate, forkIndex))
                            Debug.LogWarning("Taxi scenario could not switch routes. Schedule the cue earlier on the shared approach.", this);
                        break;
                }
            }
        }

        private void LateUpdate()
        {
            server?.Advance(AudioSettings.dspTime - audioStart);
        }

        private void OnGUI()
        {
            string status = (streamError ?? server?.Error) ?? (server?.Status switch
            {
                TaxiReplayServer.State.Waiting => "Waiting for Android on port 5005",
                TaxiReplayServer.State.Connected => "Android connected",
                TaxiReplayServer.State.Streaming => $"Android connected · {server.QueuedTicks} GPS/audio pairs",
                TaxiReplayServer.State.Finished => $"Stream complete · {server.QueuedTicks} GPS/audio pairs",
                _ => selectedScenario >= 0 && driver.HasArrived ? "Destination reached" : string.Empty
            });
            menu.Draw(gameObject.scene, scenarioIds, selectedScenario, status, LoadScenario);
        }

        private void ClearTimeline()
        {
            server?.Dispose();
            server = null;
            readyRoute = null;
            streamRide = null;
            selectedScenario = -1;
            scheduled.Clear();
            foreach (AudioSource source in audioSources)
            {
                source.Stop();
                Destroy(source);
            }
            audioSources.Clear();
            foreach (AudioClip clip in generatedClips)
                Destroy(clip);
            generatedClips.Clear();
        }
    }
}
