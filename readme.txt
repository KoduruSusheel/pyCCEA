Quick start: unzip with script, check deps, run
===============================================
lINKS FOR THE DATASETS:
AID: A scene classification dataset
https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets 
UC Merced Land Use Dataset
https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset
LC25000
https://www.kaggle.com/datasets/javaidahmadwani/lc25000/data

STEP BY STEP PROCESS:
**Open the Team 19 python project zip,inside it open the Team 19 python project folder,then open the python sem project folder in vscode and then continue....
Note (Run the steps in order)
1) Unzip datasets into data/ using the provided script

Place the zip files in the project root (recommended) with these names or adjust paths accordingly:
   - AID.zip
   - UCMerced_LandUse.zip
   - lung_colon_image_set.zip
```powershell
python .\unzipFile.py
```
''''''''''''OPTIONAL'''''''''''''''''''''
Optional flags:

```powershell
# Scan a different folder for .zip files
python .\unzipFile.py --source .\downloads

# Re-extract even if the target folder already exists
python .\unzipFile.py --force
```
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Notes:
- The script scans for all `.zip` files in the specified folder (default: project root) and extracts them into `data/`.
- Expected structures after success include:
   - `data/AID/` (classes inside) — if your AID zip contains a single root folder named `AID`
   - `data/UCMerced_LandUse/Images/` (classes inside)
   - `data/lung_colon_image_set/Train and Validation Set` and `data/lung_colon_image_set/Test Set`

2) Check and install required dependencies

```powershell
python .\check_install_deps.py
```

3) Run the model

```powershell
python .\main.py
```

When prompted, choose the dataset number:
   1: AID (data/AID)
   2: UC Merced (data/UCMerced_LandUse)
   3: LC25000 (data/lung_colon_image_set)

That's it.
Thankyou sir.