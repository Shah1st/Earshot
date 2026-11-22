using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

namespace Earshot.Simulation
{
    /// <summary>Sends one ride per connection. The caller advances it using the playback clock.</summary>
    public sealed class TaxiReplayServer : IDisposable
    {
        public enum State { Waiting, Connected, Streaming, Finished, Failed }

        private readonly TcpListener listener;
        private readonly BlockingCollection<string> messages = new BlockingCollection<string>(12);
        private readonly Thread worker;
        private volatile TcpClient client;
        private volatile State state = State.Waiting;
        private volatile bool disposed;
        private ReferenceRide ride;
        private Func<int, ReplayGpsFix> sampleGps;
        private ReplayGpsFix nextFix;
        private int nextTick;
        private bool ending;

        public State Status => state;
        public string Error { get; private set; }
        public int Port { get; }
        public int QueuedTicks => nextTick;

        public TaxiReplayServer(int port)
        {
            listener = new TcpListener(IPAddress.Loopback, port);
            listener.Start(1);
            Port = ((IPEndPoint)listener.LocalEndpoint).Port;
            worker = new Thread(Serve) { IsBackground = true, Name = "Earshot replay sender" };
            worker.Start();
        }

        public void Begin(ReferenceRide reference, Func<int, ReplayGpsFix> liveGps = null)
        {
            if (state != State.Connected || ride != null)
                throw new InvalidOperationException("Start a ride only after a receiver connects.");
            ride = reference ?? throw new ArgumentNullException(nameof(reference));
            if (ride.Gps == null && liveGps == null)
                throw new ArgumentException("This ride needs a live GPS sampler.", nameof(liveGps));
            sampleGps = liveGps;
            nextFix = sampleGps?.Invoke(0);
            state = State.Streaming;
            Queue(ride.StartMessage());
        }

        public void Advance(double playbackSeconds)
        {
            if (state != State.Streaming || ending)
                return;

            while (nextTick < ride.TickCount && playbackSeconds >= (nextTick + 1) * ReferenceRide.TickSeconds)
            {
                if (!Queue(ride.TickMessages(nextTick, nextFix)))
                    return;
                nextTick++;
                if (nextTick < ride.TickCount)
                    nextFix = sampleGps?.Invoke(nextTick);
            }

            if (nextTick == ride.TickCount && Queue(ride.StopMessage()))
            {
                ending = true;
                messages.CompleteAdding();
            }
        }

        private bool Queue(string message)
        {
            if (messages.TryAdd(message))
                return true;
            Fail("Receiver could not keep up. Reconnect and replay from the beginning.");
            client?.Close();
            return false;
        }

        private void Serve()
        {
            try
            {
                using TcpClient connected = listener.AcceptTcpClient();
                client = connected;
                if (disposed)
                    return;
                connected.NoDelay = true;
                connected.SendTimeout = 3000;
                using NetworkStream stream = connected.GetStream();
                state = State.Connected;
                foreach (string message in messages.GetConsumingEnumerable())
                {
                    if (disposed || state == State.Failed)
                        return;
                    byte[] bytes = Encoding.UTF8.GetBytes(message);
                    stream.Write(bytes, 0, bytes.Length);
                }
                if (!disposed && state != State.Failed)
                    state = State.Finished;
            }
            catch (Exception exception) when (exception is SocketException || exception is IOException
                || exception is ObjectDisposedException || exception is InvalidOperationException)
            {
                if (!disposed && state != State.Failed)
                    Fail($"Replay connection closed: {exception.Message}");
            }
            finally
            {
                listener.Stop();
            }
        }

        private void Fail(string error)
        {
            Error = error;
            state = State.Failed;
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;
            listener.Stop();
            client?.Close();
            messages.CompleteAdding();
            // Closing the listener and client releases either blocking network operation.
            if (worker.Join(100))
                messages.Dispose();
        }
    }
}
