'''
Parcellates NIFTI or CIFTI files.

NIFTI parcellation is performed via nilearn
CIFTI parcellation is performed to replicate "wb_command -cifti-parcellate .. -method MEAN"

see https://nbviewer.org/github/neurohackademy/nh2020-curriculum/blob/master/we-nibabel-markiewicz/NiBabel.ipynb

'''

import numpy as np
import nibabel as nb
import argparse
from nilearn.maskers import NiftiLabelsMasker
from nilearn.image import new_img_like

# Create parser for options
parser = argparse.ArgumentParser(
    description='''Run python based cifti-parcellate''')

# These parameters must be passed to the function
parser.add_argument('--input',
                    type=str,
                    default=None,
                    help='''Input cifti dtseries.nii or volume nii.gz file''')

parser.add_argument('--type',
                    type=str,
                    choices=['CIFTI', 'NIFTI'],
                    required=True,
                    help='''Type of input file CIFTI or NIFTI''')

parser.add_argument('--parc',
                    type=str,
                    default=None,
                    help='''Parcellation file''')

parser.add_argument('--output',
                    type=str,
                    default='output.csv',
                    help='''path to output file, e.g., output.csv''')


def surf_data_from_cifti(data, axis, surf_name):
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    for name, data_indices, model in axis.iter_structures():  # Iterates over volumetric and surface structures
        if name == surf_name:                                 # Just looking for a surface
            # Assume brainmodels axis is last, move it to front
            data = data.T[data_indices]
            # Generally 1-N, except medial wall vertices
            vtx_indices = model.vertex
            surf_data = np.zeros((vtx_indices.max() + 1,) +
                                 data.shape[1:], dtype=data.dtype)
            surf_data[vtx_indices] = data
            return surf_data
    raise ValueError(f"No structure named {surf_name}")


def volume_from_cifti(data, axis):
    assert isinstance(axis, nb.cifti2.BrainModelAxis)
    # Assume brainmodels axis is last, move it to front
    data = data.T[axis.volume_mask]
    # Which indices on this axis are for voxels?
    volmask = axis.volume_mask
    # ([x0, x1, ...], [y0, ...], [z0, ...])
    vox_indices = tuple(axis.voxel[axis.volume_mask].T)
    vol_data = np.zeros(axis.volume_shape + data.shape[1:],  # Volume + any extra dimensions
                        dtype=data.dtype)
    vol_data[vox_indices] = data                             # "Fancy indexing"
    return nb.Nifti1Image(vol_data, axis.affine)


def decompose_cifti(img):
    data = img.get_fdata(dtype=np.float32)
    brain_models = img.header.get_axis(1)  # Assume we know this
    return (volume_from_cifti(data, brain_models),
            surf_data_from_cifti(data, brain_models,
                                 "CIFTI_STRUCTURE_CORTEX_LEFT"),
            surf_data_from_cifti(data, brain_models,
                                 "CIFTI_STRUCTURE_CORTEX_RIGHT"))


def valid_parcels(arr):
    """Return unique parcel IDs excluding 0."""
    return np.setdiff1d(np.unique(arr), [0])


def parcellate_data_vectorized(data, parcels):
    """
    Compute mean time series per parcel without explicit loops.

    Parameters
    ----------
    data : ndarray
        Shape (n_vertices, n_timepoints)
    parcels : ndarray
        Parcel labels per vertex/voxel, shape (n_vertices,)

    Returns
    -------
    time_series : ndarray
        Shape (n_timepoints, n_parcels)
    parcel_ids : ndarray
        Sorted unique parcel IDs (excluding 0)
    """
    # Get unique parcel IDs (excluding 0)
    parcel_ids = np.setdiff1d(np.unique(parcels), [0])
    n_parcels = len(parcel_ids)
    n_timepoints = data.shape[1]

    # Map parcel IDs to contiguous indices
    pid_to_idx = {pid: idx for idx, pid in enumerate(parcel_ids)}
    idx_array = np.array([pid_to_idx.get(p, -1) for p in parcels])

    # Remove vertices not in any parcel
    mask = idx_array >= 0
    idx_array = idx_array[mask]
    data = data[mask]

    # Allocate array for sum and count
    sums = np.zeros((n_parcels, n_timepoints), dtype=data.dtype)
    counts = np.zeros(n_parcels, dtype=np.int32)

    # Sum data per parcel
    np.add.at(sums, idx_array, data)
    # Count number of vertices per parcel
    np.add.at(counts, idx_array, 1)

    # Divide sum by count to get mean
    # shape: (n_parcels, n_timepoints) → transpose for (time, parcels)
    time_series = (sums.T / counts).T
    return time_series.T, parcel_ids  # shape: (timepoints, parcels)


def cifti_parcellate(input_cifti, parc_cifti, output):
    '''
    Parcellate a cifti file using a parc file.
    Should replicate wb_command cifti-parcellate -method MEAN
    '''

    # load the parcellation
    assert parc_cifti.endswith('.dlabel.nii'), "Unrecognised CIFTI parc"
    pvol, pleft, pright = decompose_cifti(nb.load(parc_cifti))

    # load the timeseries (dtseries data)
    vol, left, right = decompose_cifti(nb.load(input_cifti))

    # load and flatten the volume arrays
    pvol = pvol.get_fdata().reshape(-1)
    vol = vol.get_fdata().reshape(-1, vol.shape[3])
    assert left.shape[0] == pleft.shape[0], "input and parc dims do not match!"

    # get all parcel ids across volume and surface
    parcel_ids = np.hstack(
        (np.unique(pleft), np.unique(pright), np.unique(pvol)))

    # remove '0' parcel
    parcel_ids = np.delete(parcel_ids, np.where(parcel_ids == 0)[0])

    # preallocate
    time_series = np.zeros((left.shape[1], len(parcel_ids)))

    # for each data, loop through and calculate mean.
    # loop through parcels, index unique parcel in space, avg,
    # take bold values and store in time_series
    for p in valid_parcels(pleft):
        # p-1 for pythonic indexing
        parcel_index = np.ravel(pleft) == p
        time_series[:, int(p)-1] = np.mean(left[parcel_index, :], axis=0)

    for p in valid_parcels(pright):
        # p-1 for pythonic indexing
        parcel_index = np.ravel(pright) == p
        time_series[:, int(p)-1] = np.mean(right[parcel_index, :], axis=0)

    for p in valid_parcels(pvol):
        # p-1 for pythonic indexing
        parcel_index = pvol == p
        time_series[:, int(p)-1] = np.mean(vol[parcel_index], axis=0)

    # save out
    np.savetxt(output, time_series, delimiter=',')
    return time_series


def nifti_parcellate(input, parc, output):
    masker = NiftiLabelsMasker(
        labels_img=parc, memory_level=5, verbose=1)
    time_series = masker.fit_transform(input)

    # save out
    np.savetxt(output, time_series, delimiter=',')
    return time_series


if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()

    # run parcellate
    if args.type == "NIFTI":
        nifti_parcellate(args.input, args.parc, args.output)

    elif args.type == "CIFTI":
        cifti_parcellate(args.input, args.parc, args.output)

    else:
        print("type not known")
