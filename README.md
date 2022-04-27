# [Deep Face Python Library](https://github.com/serengil/deepface) exploration in a CSharp integrated project for face detection and recognition
Obs: Branches - main for stable version and dev for development (unstable/unfinished version).

## Basic flow:
1. Enchance database: for each person photo in \Database , detect face, crop image and resize it.
```
Time to enchance database: 0.0562972 seconds.
```
2. Open camera, and start loop.
3. Retrieve camera frame.
4. Detect faces draw landmarks and show it.
<br/>![165520404-6f3682ef-da80-42b0-9637-9ae3bd17ccc1](https://user-images.githubusercontent.com/53698082/165522380-a068fab1-b1f3-4977-9587-49280754d3ae.jpg)
5. Crop detect face and save image to Temp directory.
<br/>![image](https://user-images.githubusercontent.com/53698082/165520528-962a874f-ae0b-4672-85e8-0df485523a8a.png)
7. Run Python script.
8. Python script retrieve face image from Temp directory.
9. Representation: Load existing representations if exists or encode each enchanced image using DeepFace specified model (VGG-Face by default) and save it.
```
Representing C:\Dev\Github\TiagoSanti\deep-face\Database\Images\Person 1/IMG_1783.JPEG_cropped.png..
Representing C:\Dev\Github\TiagoSanti\deep-face\Database\Images\Person 1/IMG_1784.JPEG_cropped.png..

...

Representing C:\Dev\Github\TiagoSanti\deep-face\Database\Images\Person 8/IMG_1878.JPEG_cropped.png..
Representing C:\Dev\Github\TiagoSanti\deep-face\Database\Images\Person 8/IMG_1879.JPEG_cropped.png..
Representing C:\Dev\Github\TiagoSanti\deep-face\Database\Images\Tiago Clarintino Santi/IMG_1833.JPEG_cropped.png..
Representing C:\Dev\Github\TiagoSanti\deep-face\Database\Images\Tiago Clarintino Santi/IMG_1834.JPEG_cropped.png..
Representing C:\Dev\Github\TiagoSanti\deep-face\Database\Images\Tiago Clarintino Santi/IMG_1835.JPEG_cropped.png..
Representing C:\Dev\Github\TiagoSanti\deep-face\Database\Images\Tiago Clarintino Santi/IMG_1836.JPEG_cropped.png..

Representations stored in  C:\Dev\Github\TiagoSanti\deep-face\Database\representations_vgg_face.pkl  file.
Please delete this file when you add new identities in your database.
```
11. Comparison: calculate distances between retrieved face and all database images.
13. Score: calculate scores based on distances.
```
19.62 seconds to compare -------------------------------------

                  identity  VGG-Face_cosine      score
0   Tiago Clarintino Santi         0.171098  34.159362
1   Tiago Clarintino Santi         0.192413  27.010318
2   Tiago Clarintino Santi         0.244325  16.751868
3   Tiago Clarintino Santi         0.391045   6.539524
4                 Person 8         0.409100   5.975056
5                 Person 8         0.415929   5.780445
6                 Person 5         0.416495   5.764745

...

34                Person 6         0.526000   3.614339
35                Person 6         0.526509   3.607351
```
15. KNN: get k closest matches and check which person has more closest encodings.
16. Show predict by distances mean, score mean and knn.
```
                       id  distance_mean  score_mean  knn
0  Tiago Clarintino Santi       0.249720   21.115268    4
1                Person 8       0.436408    5.324160    0
2                Person 5       0.438030    5.236152    0
3                Person 3       0.452481    4.920384    0
4                Person 7       0.478286    4.410019    0
5                Person 2       0.482048    4.322544    0
6                Person 1       0.481894    4.307662    0
7                Person 6       0.518332    3.728899    0
8                Person 4       0.518177    3.725565    0

PREDIC FILE: person_0.png
RESULT FILE: Tiago Clarintino Santi_31_638240.jpg
Result by distance mean: Tiago Clarintino Santi
Result by score mean: Tiago Clarintino Santi
Result by k_neighbors: Tiago Clarintino Santi
```
18. If all methods returns same person, save image with person name as file name, otherwise save image with each person matched by each method as file name.
<br/>![image](https://user-images.githubusercontent.com/53698082/165521281-759f576f-415c-4e1e-9d91-4e771edf45c2.png)
