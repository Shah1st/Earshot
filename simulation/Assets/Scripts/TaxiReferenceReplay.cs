using System;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Earshot.Simulation
{
    [RequireComponent(typeof(AudioSource))]
    public sealed class TaxiReferenceReplay : MonoBehaviour
    {
        [Serializable]
        private sealed class RideManifest
        {
            public int sampleRate = 0;
            public int samplesPerTick = 0;
            public RideEntry[] rides = null;
        }

        [Serializable]
        private sealed class RideEntry
        {
            public string id = null;
            public ReplayCoordinate[] route = null;
        }

        [SerializeField] private string rideId = "demo_2_roadworks";
        [SerializeField] private int port = 5005;

        private static readonly Color PlannedColor = new Color(0.32f, 0.70f, 1f);
        private static readonly Color ActualColor = new Color(0.37f, 0.95f, 0.65f);
        private ReferenceRide ride;
        private TaxiReplayServer server;
        private AudioSource player;
        private AudioClip clip;
        private Vector2[] planned;
        private Vector2[] actual;
        private Rect mapBounds;
        private double playbackStart;
        private bool started;
        private bool previousRunInBackground;
        private string startupError;

        private void Awake()
        {
            previousRunInBackground = Application.runInBackground;
            Application.runInBackground = true;
        }

        private void Start()
        {
            try
            {
                string folder = Path.Combine(Application.streamingAssetsPath, "ReferenceRides");
                var manifest = JsonUtility.FromJson<RideManifest>(File.ReadAllText(Path.Combine(folder, "replay.json")));
                if (manifest?.rides == null || manifest.sampleRate != ReferenceRide.SampleRate
                    || manifest.samplesPerTick != ReferenceRide.SamplesPerTick)
                    throw new InvalidDataException("Invalid reference ride manifest.");

                RideEntry entry = Array.Find(manifest.rides, candidate => candidate?.id == rideId);
                if (entry == null)
                    throw new InvalidDataException($"Unknown reference ride {rideId}.");

                var fixes = File.ReadLines(Path.Combine(folder, entry.id + ".gps.jsonl"))
                    .Select(JsonUtility.FromJson<ReplayGpsFix>).ToArray();
                ride = new ReferenceRide(entry.id, entry.route, fixes,
                    ReferenceRide.ReadPcmWave(Path.Combine(folder, entry.id + ".wav")));
                FitMap();

                float[] samples = ride.PlaybackSamples();
                clip = AudioClip.Create(rideId, samples.Length, 1, ReferenceRide.SampleRate, false);
                if (!clip.SetData(samples, 0))
                    throw new InvalidOperationException("Unity could not load the reference audio.");

                player = GetComponent<AudioSource>();
                player.clip = clip;
                player.loop = false;
                player.spatialBlend = 0;
                server = new TaxiReplayServer(port);
                Debug.Log($"Earshot replay waiting on 127.0.0.1:{server.Port} for {rideId}.", this);
            }
            catch (Exception exception)
            {
                startupError = exception.Message;
                Debug.LogError($"Earshot replay: {startupError}", this);
            }
        }

        private void Update()
        {
            if (server == null)
                return;

            if (server.Status == TaxiReplayServer.State.Connected)
            {
                server.Begin(ride);
                playbackStart = AudioSettings.dspTime + 0.1;
                player.PlayScheduled(playbackStart);
                started = true;
            }

            if (started && server.Status == TaxiReplayServer.State.Failed)
            {
                player.Stop();
                started = false;
            }
            if (started)
                server.Advance(AudioSettings.dspTime - playbackStart);
        }

        private void OnDestroy()
        {
            Application.runInBackground = previousRunInBackground;
            server?.Dispose();
            if (clip != null)
                Destroy(clip);
        }

        private void FitMap()
        {
            double baseLat = ride.Route[0].lat;
            double baseLon = ride.Route[0].lon;
            double longitudeScale = Math.Cos(baseLat * Math.PI / 180);
            Vector2 Project(double lat, double lon) =>
                new Vector2((float)((lon - baseLon) * longitudeScale), (float)(baseLat - lat));

            planned = ride.Route.Select(point => Project(point.lat, point.lon)).ToArray();
            actual = ride.Gps.Select(point => Project(point.lat, point.lon)).ToArray();
            Vector2 min = planned[0];
            Vector2 max = min;
            foreach (Vector2 point in planned.Concat(actual))
            {
                min = Vector2.Min(min, point);
                max = Vector2.Max(max, point);
            }
            mapBounds = new Rect(min, Vector2.Max(max - min, Vector2.one * 0.00001f));
        }

        private void OnGUI()
        {
            GUILayout.BeginArea(new Rect(16, 16, Mathf.Max(1, Screen.width - 32), 100));
            GUILayout.Label($"Earshot / {rideId}");
            GUILayout.Label(startupError ?? server?.Error ?? ConnectionStatus());
            if (server != null && server.QueuedTicks > 0)
            {
                int tick = server.QueuedTicks - 1;
                ReplayGpsFix fix = ride.Gps[tick];
                GUILayout.Label($"{(tick + 1) * ReferenceRide.TickSeconds:0.0} s   |   "
                    + $"{fix.speed_kmh:0} km/h   |   {fix.lat:F5}, {fix.lon:F5}");
            }
            GUILayout.EndArea();

            if (planned == null || Event.current.type != EventType.Repaint || Screen.height < 160 || Screen.width < 120)
                return;

            var frame = new Rect(26, 122, Screen.width - 52, Screen.height - 148);
            float scale = Mathf.Min(frame.width / mapBounds.width, frame.height / mapBounds.height);
            Vector2 Plot(Vector2 point) => frame.center + (point - mapBounds.center) * scale;
            for (int i = 1; i < planned.Length; i++)
                DrawLine(Plot(planned[i - 1]), Plot(planned[i]), PlannedColor, 5);

            int count = server?.QueuedTicks ?? 0;
            for (int i = 1; i < count; i++)
                DrawLine(Plot(actual[i - 1]), Plot(actual[i]), ActualColor, 2);
            if (count > 0)
            {
                Vector2 car = Plot(actual[count - 1]);
                GUI.DrawTexture(new Rect(car.x - 4, car.y - 4, 8, 8), Texture2D.whiteTexture);
            }
            GUI.Label(new Rect(20, 92, 100, 22), "<color=#52b3ff>Planned route</color>");
            GUI.Label(new Rect(120, 92, 100, 22), "<color=#5ef2a6>Actual route</color>");
        }

        private string ConnectionStatus()
        {
            switch (server?.Status)
            {
                case TaxiReplayServer.State.Waiting: return $"Waiting for Android or a test receiver on port {server.Port}";
                case TaxiReplayServer.State.Connected: return "Receiver connected";
                case TaxiReplayServer.State.Streaming: return "Streaming GPS and audio";
                case TaxiReplayServer.State.Finished: return "Ride complete. Restart Play Mode for another connection.";
                default: return string.Empty;
            }
        }

        private static void DrawLine(Vector2 from, Vector2 to, Color color, float thickness)
        {
            Vector2 delta = to - from;
            if (delta.sqrMagnitude < 0.01f)
                return;
            Matrix4x4 savedMatrix = GUI.matrix;
            Color savedColor = GUI.color;
            GUI.color = color;
            GUIUtility.RotateAroundPivot(Mathf.Atan2(delta.y, delta.x) * Mathf.Rad2Deg, from);
            GUI.DrawTexture(new Rect(from.x, from.y - thickness / 2, delta.magnitude, thickness), Texture2D.whiteTexture);
            GUI.matrix = savedMatrix;
            GUI.color = savedColor;
        }
    }
}
