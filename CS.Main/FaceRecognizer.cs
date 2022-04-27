namespace DeepFace
{
    public class FaceRecognizer
    {
        public Task PyTask;
        public PythonScript PyScript;
        public string[] Args = CreatePythonScript();

        public FaceRecognizer()
        {
            PyScript = new PythonScript(Args);
            PyTask = new(PyScript.StartProcess);
        }

        private static string[] CreatePythonScript()
        {
            string projectPath = Helpers.GetProjectPath();
            string scriptPath = projectPath + "\\PyScripts\\verification.py";
            string databasePath = projectPath + "\\Database";
            string tempPath = projectPath + "\\Temp";
            string[] args = { scriptPath, databasePath, tempPath };

            return args;
        }

        public void RunPyTask()
        {
            PyTask.Start();
        }

        public void ReloadAndRunPyTask()
        {
            PyScript.CreateProcess();
            PyTask = new(PyScript.StartProcess);
        }

        public TaskStatus? GetPyTaskStatus()
        {
            if (PyTask != null)
            {
                return PyTask.Status;
            }
            else
            {
                return null;
            }
        }
    }
}
