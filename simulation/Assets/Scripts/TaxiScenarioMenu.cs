using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace Earshot.Simulation
{
    internal sealed class TaxiScenarioMenu
    {
        public const int PanelWidth = 288;
        private GUIStyle title;
        private GUIStyle message;
        private bool reloading;
        private string reloadError;

        public void Draw(Scene scene, IReadOnlyList<string> scenarios, int selected, string status, Action<int> select)
        {
            title ??= new GUIStyle(GUI.skin.label) { fontStyle = FontStyle.Bold, fontSize = 15 };
            message ??= new GUIStyle(GUI.skin.label) { wordWrap = true };
            float width = Mathf.Max(1, Mathf.Min(PanelWidth, Screen.width - 32));
            float left = Screen.width - width - 16;
            GUILayout.BeginArea(new Rect(left, 16, width, Mathf.Max(1, Screen.height - 32)));
            GUILayout.BeginVertical(GUI.skin.box);
            GUILayout.Label(selected >= 0 ? DisplayName(scenarios[selected]) : "Choose a taxi scenario", title);
            if (!string.IsNullOrEmpty(status))
                GUILayout.Label(status, message);
            if (!string.IsNullOrEmpty(reloadError))
                GUILayout.Label(reloadError, message);

            bool previous = GUI.enabled;
            GUI.enabled = previous && !reloading;
            if (selected < 0)
            {
                for (int i = 0; i < scenarios.Count; i++)
                {
                    if (GUILayout.Button(DisplayName(scenarios[i]), GUILayout.MinHeight(30)))
                        select(i);
                }
            }
            GUILayout.Space(4);
            if (GUILayout.Button(reloading ? "Reloading…" : "Reload scene", GUILayout.MinHeight(30)))
                Reload(scene);
            GUI.enabled = previous;
            GUILayout.EndVertical();
            GUILayout.EndArea();
        }

        private void Reload(Scene scene)
        {
            try
            {
                reloading = SceneManager.LoadSceneAsync(scene.path, LoadSceneMode.Single) != null;
                if (!reloading)
                    reloadError = "Could not reload this scene.";
            }
            catch (Exception error)
            {
                reloadError = $"Could not reload this scene. {error.Message}";
            }
        }

        private static string DisplayName(string id)
        {
            string[] parts = id.Split(new[] { '_' }, 3);
            string name = parts.Length == 3 && parts[0] == "demo" ? parts[2] : id;
            name = name.Replace('_', ' ');
            if (name == "normal")
                return "Normal ride";
            return name.Length == 0 ? "Unnamed scenario" : char.ToUpperInvariant(name[0]) + name.Substring(1);
        }
    }
}
