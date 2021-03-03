# DeepMRAC_headneck

Deep learning 3D-U-net for deriving synthetic CT from Dixon-MRI (in-phase and opposed phase) images.

## Requirements
Install appropriate GPU, CUDA and cuDNN toolkits. See https://www.tensorflow.org/install/gpu.

To install the required software tools using pip, run:
```
pip install -r requirements.txt
```
The procssing of data and running the script currently rely on the minc toolkit (https://github.com/BIC-MNI/minc-toolkit-v2)

## Prepare data

Dicom data is converted into minc files 
```
dcm2mnc ﹤path to DICOM data﹥ ﹤path to output﹥ -dname '' -fname ﹤output filename﹥ -usecoordinates
```
The Dixon-MRI images (in-phase and opposed phase) are resampled and normalized by the following:
```
autocrop -isostep 1.3021 input.mnc output1.mnc -clobber
mincreshape -dimorder zspace,yspace,xspace -dimrange xspace=-16,416 -dimrange yspace=11,288 output1.mnc output2.mnc -clobber
sd=`volume_stats -stddev output2.mnc -quiet -floor 0.000001`; 
mean=`volume_stats -mean output2.mnc -quiet -floor 0.000001`; 
minccalc -expression "(A[0]-$mean)/$sd;" output2.mnc output.mnc -clobber;
```

## Predicting synthetic CT (running script)

The synthetic CT is derived by the following command 
```
python3 predict.py ﹤path to trained model (avaiable upon request)﹥ ﹤path to patient data﹥ ﹤name of Dixon in-phase file (.mnc)﹥ ﹤name of Dixon opposed-phase file (.mnc)﹥ ﹤path to output﹥ ﹤name of output file (.mnc)﹥
```

## Contact
Anders Olin, Rigshospitalet, Copenhagen, Denmark
anders.olin@regionh.dk
