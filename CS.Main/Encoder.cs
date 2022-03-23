namespace DeepFace
{
    public class Encoder
    {
        public static void EncodeDatabase()
        {
            string projectPath = Helpers.GetProjectPath();
            string scriptPath = projectPath + "\\scripts\\database_encoder.py";
            string databasePath = projectPath + "\\database";
            string[] args = {scriptPath, databasePath};

            PythonScript script = new(args);
            script.StartProcess();
        }
    }
}
