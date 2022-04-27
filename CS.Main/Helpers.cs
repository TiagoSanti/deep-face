using OpenCvSharp;
using OpenCvSharp.Extensions;
using System.Drawing;
using Img = System.Drawing.Image;
using UltraFaceDotNet;
using System.Text;
using System.Globalization;

namespace DeepFace
{
    public class Helpers
    {
        #region Converters
        public static Bitmap MatToBitmap(Mat mat)
        {
            try
            {
                return BitmapConverter.ToBitmap(mat);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return null;
            }
        }

        public static List<Bitmap> MatsToBitmaps(List<Mat> mats)
        {
            List<Bitmap> bitmaps = new();
            foreach (Mat mat in mats)
            {
                bitmaps.Add(BitmapConverter.ToBitmap(mat));
            }
            return bitmaps;
        }

        public static List<Mat> BitmapsToMats(List<Bitmap> bitmaps)
        {
            List<Mat> mats = new();
            foreach (Bitmap bitmap in bitmaps)
            {
                mats.Add(BitmapConverter.ToMat(bitmap));
            }
            return mats;
        }

        #endregion

        #region Image Manager
        public static Mat ResizeImageFromPath(string imageFilePath)
        {
            OpenCvSharp.Size size;
            Mat mat = Cv2.ImRead(imageFilePath);
            if (mat.Width > mat.Height)
            {
                size = new(320, 240);
            }
            else
            {
                size = new(240, 320);
            }

            Mat mat_resize = new(size, MatType.CV_16UC3);

            Cv2.Resize(mat, mat_resize, size, 0, 0, InterpolationFlags.LinearExact);

            return mat_resize;
        }

        public static List<Mat> ResizeImagesFromMats(List<Mat> mats)
        {
            List<Mat> mats_resize = new();
            OpenCvSharp.Size size;

            foreach (Mat mat in mats)
            {
                if (mat.Width > mat.Height)
                {
                    size = new(320, 240);
                }
                else
                {
                    size = new(240, 320);
                }

                Mat mat_resize = new(size, MatType.CV_16UC3);

                Cv2.Resize(mat, mat_resize, size, 0, 0, InterpolationFlags.LinearExact);

                mats_resize.Add(mat_resize);
            }
            
            return mats_resize;
        }

        public static Bitmap CropImageFromPath(Mat mat, FaceInfo faceInfo)
        {
            Bitmap bitmap = MatToBitmap(mat);

            int X1 = (int)faceInfo.X1;
            int Y1 = (int)faceInfo.Y1;
            int width = (int)(faceInfo.X2 - faceInfo.X1);
            int height = (int)(faceInfo.Y2 - faceInfo.Y1);

            Rectangle faceRectangle = new(X1, Y1, width, height);

            Bitmap cropped = new(faceRectangle.Width, faceRectangle.Height);
            
            using Graphics g = Graphics.FromImage(cropped);
            g.DrawImage(bitmap, -faceRectangle.X, -faceRectangle.Y);

            return cropped;
        }

        public static List<Bitmap> CropImageFromMat(Mat mat, FaceInfo[] faceInfos)
        {
            using Bitmap bitmap = MatToBitmap(mat);
            List<Bitmap> croppeds = new();

            foreach (FaceInfo faceInfo in faceInfos)
            {
                int X1 = (int)faceInfo.X1;
                int Y1 = (int)faceInfo.Y1;
                int width = (int)(faceInfo.X2 - faceInfo.X1);
                int height = (int)(faceInfo.Y2 - faceInfo.Y1);

                Rectangle faceRectangle = new(X1, Y1, width, height);

                Bitmap cropped = new(faceRectangle.Width, faceRectangle.Height);
                using Graphics g = Graphics.FromImage(cropped);
                g.DrawImage(bitmap, -faceRectangle.X, -faceRectangle.Y);

                croppeds.Add(cropped);
            }

            return croppeds;
        }

        internal static void OverwriteImage(Bitmap bitmap, string personImagePath)
        {
            File.Delete(personImagePath);
            bitmap.Save(personImagePath+"_cropped.png", System.Drawing.Imaging.ImageFormat.Png);
        }

        public static void SaveTempImage(List<Mat> mats)
        {
            string tempPath = GetProjectPath() + "\\temp\\";
            if (!Directory.Exists(tempPath))
            {
                Directory.CreateDirectory(tempPath);
            }

            int i = 0;
            foreach (Mat mat in mats)
            {
                var fileName = "person_"+i;
                Cv2.ImWrite(tempPath + fileName + ".png", mat);
                i++;
            }
        }
        #endregion

        #region Path Manager
        public static string GetProjectPath()
        {
            string current = Directory.GetCurrentDirectory();   // \net6.0
            current = Directory.GetParent(current).FullName;    // \Debug
            current = Directory.GetParent(current).FullName;    // \x64
            current = Directory.GetParent(current).FullName;    // \bin
            current = Directory.GetParent(current).FullName;    // \UltraFaceRecognition
            return current;
        }

        public static void CreateTempDir()
        {
            string tempPath = GetProjectPath() + "\\Temp";
            if (!Directory.Exists(tempPath))
            {
                Directory.CreateDirectory(tempPath);
            }
        }

        public static bool ThereIsTempFiles()
        {
            string tempPath = GetProjectPath() + "\\temp";
            string[] tempImages = Directory.GetFiles(tempPath);

            return tempImages.Length > 0;
        }

        public static void ClearTemp()
        {
            if (ThereIsTempFiles())
            {
                string[] files = Directory.GetFiles(GetProjectPath() + "\\Temp");
                foreach (string file in files)
                {
                    File.Delete(file);
                }
            }
        }
        #endregion

        #region Formatter
        public static string StringNormalization(string formattedString)
        {
            if (formattedString != null)
            {
                formattedString = formattedString.ToUpper();
                formattedString = RemoveAccentsAndCedilla(formattedString);
            }
            return formattedString;
        }

        public static string RemoveAccentsAndCedilla(string formattedString)
        {
            return new string(formattedString
             .Normalize(NormalizationForm.FormD)
             .Where(ch => char.GetUnicodeCategory(ch) != UnicodeCategory.NonSpacingMark && !"´~^¸¨".Contains(ch))
             .ToArray());
        }
        #endregion
    }
}