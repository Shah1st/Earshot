using System;
using UnityEngine;

namespace Earshot.Simulation
{
    public readonly struct TaxiGpsCoordinate
    {
        public double Latitude { get; }
        public double Longitude { get; }

        public TaxiGpsCoordinate(double latitude, double longitude)
        {
            Latitude = latitude;
            Longitude = longitude;
        }
    }

    /// <summary>Maps the flat city to a small geographic area using a local WGS84 approximation.</summary>
    [Serializable]
    public sealed class TaxiGpsReference
    {
        [Tooltip("City position that corresponds to the reference latitude and longitude.")]
        [SerializeField] private Vector3 origin = new Vector3(-305.7f, 0f, -114.6f);
        [SerializeField, Range(-89f, 89f)] private double latitude = 46.51910;
        [SerializeField, Range(-180f, 180f)] private double longitude = 6.56680;

        /// <summary>One world unit is a metre. +X points east, +Z points north.</summary>
        public TaxiGpsCoordinate Project(Vector3 worldPosition)
        {
            const double semiMajorAxis = 6378137.0;
            const double eccentricitySquared = 0.0066943799901413165;
            const double radiansToDegrees = 180.0 / Math.PI;

            double referenceLatitude = latitude / radiansToDegrees;
            double sinLatitude = Math.Sin(referenceLatitude);
            double denominator = 1.0 - eccentricitySquared * sinLatitude * sinLatitude;
            double primeVerticalRadius = semiMajorAxis / Math.Sqrt(denominator);
            double meridianRadius = primeVerticalRadius * (1.0 - eccentricitySquared) / denominator;
            double parallelRadius = primeVerticalRadius * Math.Cos(referenceLatitude);

            double north = (double)worldPosition.z - origin.z;
            double east = (double)worldPosition.x - origin.x;
            double projectedLatitude = latitude + north / meridianRadius * radiansToDegrees;
            double projectedLongitude = longitude + east / parallelRadius * radiansToDegrees;
            return new TaxiGpsCoordinate(projectedLatitude, Math.IEEERemainder(projectedLongitude, 360.0));
        }
    }
}
