using OpenCvSharp;
using UltraFaceDotNet;

namespace DeepFace
{
    public class Program
    {
        public static void Main()
        {
            FaceDetector detector = new();
            Helpers.CreateTempDir();
            Helpers.ClearTemp();

            var watch = System.Diagnostics.Stopwatch.StartNew();
            detector.EnchanceDatabaseImages();
            Console.WriteLine("Time to enchance database: " + watch.Elapsed.TotalSeconds.ToString() + " seconds.");

            RunRealTimeRecognizer(detector);
        }

        public static void RunRealTimeRecognizer(FaceDetector detector)
        {
            FaceRecognizer faceRecognizer = new();
            Camera capture = new();
            capture.StartCamera();

            while (Window.WaitKey(10) != 27)
            {
                using Mat? mat = capture.GetFrame();
                if (mat != null)
                {
                    FaceInfo[] faceInfos = detector.DetectFacesMat(mat);
                    if (faceInfos.Length > 0)
                    {
                        Drawers.DrawFacesRects(mat, faceInfos);

                        if (faceRecognizer.GetPyTaskStatus() == TaskStatus.Created)
                        {
                            Console.WriteLine("Initiating PyScript");
                            faceRecognizer.RunPyTask();
                        }
                        else if (faceRecognizer.GetPyTaskStatus() == TaskStatus.Running)
                        {
                            if (!Helpers.ThereIsTempFiles())
                            {
                                List<Mat> croppedMats = Helpers.BitmapsToMats(Helpers.CropImageFromMat(mat, faceInfos));
                                croppedMats = Helpers.ResizeImagesFromMats(croppedMats);
                                Helpers.SaveTempImage(croppedMats);
                            }
                        }
                    }

                    capture.ShowImage(mat);
                }
            }

            Helpers.ClearTemp();
            Camera.Close();
        }
    }
}
