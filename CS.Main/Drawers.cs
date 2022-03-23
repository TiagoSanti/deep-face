using OpenCvSharp;
using UltraFaceDotNet;

namespace DeepFace
{
    public class Drawers
    {
        public static void DrawFacesRects(Mat mat, FaceInfo[] faceInfos)
        {
            foreach (FaceInfo faceInfo in faceInfos)
            {
                OpenCvSharp.Point pt1 = new(faceInfo.X1, faceInfo.Y1);
                OpenCvSharp.Point pt2 = new(faceInfo.X2, faceInfo.Y2);

                DrawRect(mat, pt1, pt2);
            }
        }
        public static void DrawRect(Mat mat, OpenCvSharp.Point pt1, OpenCvSharp.Point pt2)
        {
            Cv2.Rectangle(mat, pt1, pt2, Scalar.Red,2);
        }
    }
}
