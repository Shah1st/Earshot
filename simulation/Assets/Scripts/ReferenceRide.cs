using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

namespace Earshot.Simulation
{
    [Serializable]
    public sealed class ReplayCoordinate
    {
        public double lat;
        public double lon;
    }

    [Serializable]
    public sealed class ReplayGpsFix
    {
        public double t;
        public double lat;
        public double lon;
        public double speed_kmh;
    }

    /// <summary>Validated data for one reference ride. Audio packets use the original PCM bytes.</summary>
    public sealed class ReferenceRide
    {
        public const int SampleRate = 16000;
        public const int SamplesPerTick = 8000;
        public const double TickSeconds = (double)SamplesPerTick / SampleRate;
        private const int BytesPerTick = SamplesPerTick * sizeof(short);

        private readonly byte[] pcm;
        public string Id { get; }
        public ReplayCoordinate[] Route { get; }
        public ReplayGpsFix[] Gps { get; }
        public int TickCount => pcm.Length / BytesPerTick;

        public ReferenceRide(string id, ReplayCoordinate[] route, ReplayGpsFix[] gps, byte[] pcm)
            : this(id, route, pcm)
        {
            if (gps == null || gps.Length != TickCount)
                throw new InvalidDataException("The ride must have one GPS record per complete audio chunk.");
            for (int i = 0; i < gps.Length; i++)
                ValidateFix(i, gps[i]);
            Gps = gps;
        }

        public ReferenceRide(string id, ReplayCoordinate[] route, byte[] pcm)
        {
            if (string.IsNullOrEmpty(id) || !Regex.IsMatch(id, "\\A[a-zA-Z0-9_-]+\\z"))
                throw new InvalidDataException("The ride ID must contain only letters, digits, hyphens or underscores.");
            if (route == null || route.Length < 2 || route.Any(p => p == null || !ValidLocation(p.lat, p.lon)))
                throw new InvalidDataException("The ride needs a valid planned route.");
            if (pcm == null || pcm.Length == 0 || pcm.Length % BytesPerTick != 0)
                throw new InvalidDataException("The ride must contain complete audio chunks.");

            Id = id;
            Route = route;
            this.pcm = pcm;
        }

        public static byte[] ReadPcmWave(string path)
        {
            using var input = new BinaryReader(File.OpenRead(path));
            if (input.BaseStream.Length < 12 || ReadTag(input) != "RIFF")
                throw new InvalidDataException("Expected a RIFF WAV file.");
            long end = 8L + input.ReadUInt32();
            if (end < 12 || end > input.BaseStream.Length || ReadTag(input) != "WAVE")
                throw new InvalidDataException("Invalid WAVE header.");

            bool formatFound = false;
            byte[] samples = null;
            while (input.BaseStream.Position + 8 <= end)
            {
                string tag = ReadTag(input);
                uint length = input.ReadUInt32();
                long next = input.BaseStream.Position + length + (length & 1);
                if (next > end)
                    throw new InvalidDataException($"Truncated WAV chunk: {tag}.");

                if (tag == "fmt ")
                {
                    if (length < 16 || input.ReadUInt16() != 1 || input.ReadUInt16() != 1
                        || input.ReadUInt32() != SampleRate || input.ReadUInt32() != SampleRate * sizeof(short)
                        || input.ReadUInt16() != sizeof(short) || input.ReadUInt16() != 16)
                        throw new InvalidDataException("Expected 16 kHz mono PCM16 audio.");
                    formatFound = true;
                }
                else if (tag == "data")
                {
                    if (samples != null || length == 0 || length > int.MaxValue || length % BytesPerTick != 0)
                        throw new InvalidDataException("The WAV data must contain whole half-second chunks.");
                    samples = input.ReadBytes((int)length);
                }
                input.BaseStream.Position = next;
            }

            if (!formatFound || samples == null)
                throw new InvalidDataException("WAV format or sample data is missing.");
            return samples;
        }

        public float[] PlaybackSamples() => PlaybackSamples(pcm);

        internal static float[] PlaybackSamples(byte[] pcm)
        {
            var samples = new float[pcm.Length / sizeof(short)];
            for (int i = 0; i < samples.Length; i++)
                samples[i] = unchecked((short)(pcm[2 * i] | (pcm[2 * i + 1] << 8))) / 32768f;
            return samples;
        }

        public string StartMessage() => "{\"type\":\"start\",\"ride_id\":\"" + Id + "\",\"route\":["
            + string.Join(",", Route.Select(p => $"[{Number(p.lat)},{Number(p.lon)}]")) + "]}\n";

        public string TickMessages(int tick, ReplayGpsFix liveFix = null)
        {
            ReplayGpsFix fix = liveFix ?? Gps?[tick];
            ValidateFix(tick, fix);
            string time = Number(fix.t);
            string data = Convert.ToBase64String(pcm, tick * BytesPerTick, BytesPerTick);
            return $"{{\"type\":\"gps\",\"t\":{time},\"lat\":{Number(fix.lat)},\"lon\":{Number(fix.lon)},\"speed_kmh\":{Number(fix.speed_kmh)}}}\n"
                + $"{{\"type\":\"audio\",\"t\":{time},\"sr\":{SampleRate},\"pcm16_b64\":\"{data}\"}}\n";
        }

        public string StopMessage() => "{\"type\":\"stop\",\"ride_id\":\"" + Id + "\"}\n";

        private static void ValidateFix(int tick, ReplayGpsFix fix)
        {
            if (fix == null || fix.t != tick * TickSeconds || !ValidLocation(fix.lat, fix.lon)
                || !Finite(fix.speed_kmh) || fix.speed_kmh < 0)
                throw new InvalidDataException($"Invalid GPS record at tick {tick}.");
        }

        private static string Number(double value) => value.ToString("R", CultureInfo.InvariantCulture);
        private static string ReadTag(BinaryReader input) => Encoding.ASCII.GetString(input.ReadBytes(4));
        private static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
        private static bool ValidLocation(double lat, double lon) => lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
    }
}
