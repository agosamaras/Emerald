import pydicom
import cv2

def return_information(dcm_file):
    # Safely extract attributes with default values
    infos_1 = f'''
    Image type: {getattr(dcm_file, 'ImageType', 'Not Available')} \n
    SOP Class UID: {getattr(dcm_file, 'SOPClassUID', 'Not Available')}\n
    SOP Instance UID: {getattr(dcm_file, 'SOPInstanceUID', 'Not Available')}\n
    Study Date: {getattr(dcm_file, 'StudyDate', 'Not Available')}\n
    Series Date: {getattr(dcm_file, 'SeriesDate', 'Not Available')}\n
    Content Date: {getattr(dcm_file, 'ContentDate', 'Not Available')}\n
    Study Time: {getattr(dcm_file, 'StudyTime', 'Not Available')}\n
    Series Time: {getattr(dcm_file, 'SeriesTime', 'Not Available')}\n
    Content Time: {getattr(dcm_file, 'ContentTime', 'Not Available')}\n
    Accession Number: {getattr(dcm_file, 'AccessionNumber', 'Not Available')}\n
    Modality: {getattr(dcm_file, 'Modality', 'Not Available')}\n
    Manufacturer: {getattr(dcm_file, 'Manufacturer', 'Not Available')}\n
    Referring Physician Name: {getattr(dcm_file, 'ReferringPhysicianName', 'Not Available')}\n
    Patient Name: {getattr(dcm_file, 'PatientName', 'Not Available')}\n
    Patient ID: {getattr(dcm_file, 'PatientID', 'Not Available')}\n
    Patient Birth Date: {getattr(dcm_file, 'PatientBirthDate', 'Not Available')}\n
    Patient Sex: {getattr(dcm_file, 'PatientSex', 'Not Available')}\n
    Slice Thickness: {getattr(dcm_file, 'SliceThickness', 'Not Available')}\n
    Patient Position: {getattr(dcm_file, 'PatientPosition', 'Not Available')}\n
    Study Instance UID: {getattr(dcm_file, 'StudyInstanceUID', 'Not Available')}\n
    Series Instance UID: {getattr(dcm_file, 'SeriesInstanceUID', 'Not Available')}\n
    '''

    infos_2 = f'''
    Study ID: {getattr(dcm_file, 'StudyID', 'Not Available')}\n
    Series Number: {getattr(dcm_file, 'SeriesNumber', 'Not Available')}\n
    Instance Number: {getattr(dcm_file, 'InstanceNumber', 'Not Available')}\n
    Image Position Patient: {getattr(dcm_file, 'ImagePositionPatient', 'Not Available')}\n
    Image Orientation Patient: {getattr(dcm_file, 'ImageOrientationPatient', 'Not Available')}\n
    Frame Of Reference UID: {getattr(dcm_file, 'FrameOfReferenceUID', 'Not Available')}\n
    Position Reference Indicator: {getattr(dcm_file, 'PositionReferenceIndicator', 'Not Available')}\n
    Samples Per Pixel: {getattr(dcm_file, 'SamplesPerPixel', 'Not Available')}\n
    Photometric Interpretation: {getattr(dcm_file, 'PhotometricInterpretation', 'Not Available')}\n
    Rows: {getattr(dcm_file, 'Rows', 'Not Available')}\n
    Columns: {getattr(dcm_file, 'Columns', 'Not Available')}\n
    Pixel Spacing: {getattr(dcm_file, 'PixelSpacing', 'Not Available')}\n
    Bits Allocated: {getattr(dcm_file, 'BitsAllocated', 'Not Available')}\n
    Bits Stored: {getattr(dcm_file, 'BitsStored', 'Not Available')}\n
    High Bit: {getattr(dcm_file, 'HighBit', 'Not Available')}\n
    Pixel Representation: {getattr(dcm_file, 'PixelRepresentation', 'Not Available')}\n
    Window Center: {getattr(dcm_file, 'WindowCenter', 'Not Available')}\n
    Window Width: {getattr(dcm_file, 'WindowWidth', 'Not Available')}\n
    Rescale Intercept: {getattr(dcm_file, 'RescaleIntercept', 'Not Available')}\n
    Rescale Slope: {getattr(dcm_file, 'RescaleSlope', 'Not Available')}\n
    RescaleType: {getattr(dcm_file, 'RescaleType', 'Not Available')}\n
    '''

    return infos_1, infos_2


def anonymize_case(dcm_file):
    dicom_file_data = pydicom.dcmread(dcm_file)

    dicom_file_data.PatientName = ''
    dicom_file_data.PatientBirthDate =''
    dicom_file_data.StudyID = ''
    dicom_file_data.PatientSex = ''

    return dicom_file_data

