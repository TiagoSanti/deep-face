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
            DatabaseEncodings(detector);
            watch.Stop();
            Console.WriteLine("Time to run DatabaseEncodings(): " + watch.Elapsed.TotalSeconds.ToString());

            RunRealTimeRecognizer();
        }

        public static void RunRealTimeRecognizer()
        {
            FaceDetector detector = new();
            FaceRecognizer faceRecognizer = new();
            Camera capture = new();
            capture.StartCamera();

            var watch = System.Diagnostics.Stopwatch.StartNew();
            while (Window.WaitKey(10) != 27)
            {
                using Mat? mat = capture.GetFrame();
                if (mat != null)
                {
                    FaceInfo[] faceInfos = detector.DetectFacesMat(mat);
                    if (faceInfos.Length > 0)
                    {
                        IEnumerable<FaceInfo> validFaceInfos = from faceInfo in faceInfos
                                                               where faceInfo.Score > 0.9
                                                               select faceInfo;
                        Drawers.DrawFacesRects(mat, faceInfos);

                        if (watch.Elapsed.TotalSeconds > 3)
                        {
                            if (faceRecognizer.GetPyTaskStatus() == TaskStatus.Created)
                            {
                                faceRecognizer.RunPyTask();
                            }
                            else if (faceRecognizer.GetPyTaskStatus() == TaskStatus.Running)
                            {
                                if (!Helpers.ThereIsTempFiles())
                                {
                                    List<Mat> croppedMats = Helpers.BitmapsToMats(Helpers.CropImageFromMat(mat, faceInfos));
                                    Helpers.SaveTempImage(croppedMats);
                                }
                            }

                            watch.Restart();
                        }
                    }

                    capture.ShowImage(mat);
                }
            }
            watch.Stop();
            Helpers.ClearTemp();
            Camera.Close();
        }

        public static void DatabaseEncodings(FaceDetector detector)
        {
            detector.EnchanceDatabaseImages();
            Encoder.EncodeDatabase();
        }
    }
}
