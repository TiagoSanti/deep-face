using OpenCvSharp;
using System.Drawing;
using UltraFaceDotNet;

namespace DeepFace
{
    public class FaceDetector
    {
        public UltraFace? ultraFace;
        public string binPath = Helpers.GetProjectPath()+@"\CS.Main\ncnn\data\version-RFB\RFB-320.bin";
        public string paramPath = Helpers.GetProjectPath()+@"\CS.Main\ncnn\data\version-RFB\RFB-320.param";
        public UltraFaceParameter? parameter;

        public FaceDetector()
        {
            parameter = new UltraFaceParameter
            {
                BinFilePath = binPath,
                ParamFilePath = paramPath,
                InputWidth = 320,
                InputLength = 240,
                NumThread = 1,
                ScoreThreshold = 0.95f
            };

            ultraFace = UltraFace.Create(parameter);
        }

        public FaceInfo[] DetectFacesMat(Mat mat)
        {
            using var inMat = NcnnDotNet.Mat.FromPixels(mat.Data, NcnnDotNet.PixelType.Bgr2Rgb, mat.Cols, mat.Rows);
            
            return ultraFace.Detect(inMat).ToArray();
        }

        public FaceInfo[] DetectFacesImagePath(string imagePath)
        {
            using var frame = Cv2.ImRead(imagePath);
            using var inMat = NcnnDotNet.Mat.FromPixels(frame.Data, NcnnDotNet.PixelType.Bgr2Rgb, frame.Cols, frame.Rows);

            return ultraFace.Detect(inMat).ToArray();
        }

        public void EnchanceDatabaseImages()
        {
            string projectDir = Helpers.GetProjectPath();
            string imageDatabaseDir = projectDir+@"\Database\Images\";

            string[] peopleImageDBDir = Directory.GetDirectories(imageDatabaseDir);
            foreach (string personImageDBDir in peopleImageDBDir)
            {
                string[] personImagesPath = Directory.GetFiles(personImageDBDir);
                foreach (string personImagePath in personImagesPath)
                {
                    if (!personImagePath.Contains("_cropped"))
                    {
                        Mat mat_resized = Helpers.ResizeImageFromPath(personImagePath);
                        FaceInfo[] faceInfos = DetectFacesMat(mat_resized);
                        if (faceInfos != null)
                        {
                            FaceInfo faceInfo = faceInfos[0];
                            using Bitmap bitmap = Helpers.CropImageFromPath(mat_resized, faceInfo);
                            Helpers.OverwriteImage(bitmap, personImagePath);
                        }
                    }
                }
            }
        }
    }
}
