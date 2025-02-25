import dicom2nifti
import dicom2nifti.settings as settings
import os
# Feed path to the downloaded data here
DATAPATH='./CHAOS_Train_Sets/Train_Sets/MR' # please put chaos dataset training fold here which contains ground truth

# Feed path to the output folder here
OUTPATH='./niis'

# settings.disable_validate_orthogonal()
# settings.enable_resampling()
# settings.set_resample_spline_interpolation_order(1)
# settings.set_resample_padding(-1000)

for sid in os.listdir(DATAPATH):
    dicom_directory =  f"{DATAPATH}/{sid}/T1DUAL/DICOM_anon/OutPhase"
    output_folder = f"{DATAPATH}/{sid}/T1DUAL/DICOM_anon/OutPhase"
    dicom2nifti.convert_directory(dicom_directory, output_folder)
