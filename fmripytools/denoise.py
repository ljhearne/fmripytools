"""
Post-fmriprep denoising using Nilearn.

Uses high-level nilearn functions to clean timeseries data.

This approach and the possible strategies are from this paper:
see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10153168/pdf/nihpp-2023.04.18.537240v3.pdf

The specific 'out-of-the-box' denoise strategies are found here:
https://github.com/SIMEXP/fmriprep-denoise-benchmark/blob/b9d44504384b3641dbd1d063105cb6eb99713488/fmriprep_denoise/dataset/benchmark_strategies.json#L4

"""
import json
import os
import nibabel as nb
from nilearn.signal import clean
from nilearn.image import clean_img
import numpy as np
import argparse
from nilearn.interfaces.fmriprep import load_confounds_strategy


# Create parser for options
parser = argparse.ArgumentParser(
    description='''Run nilearn based BOLD denoising''')

# These parameters must be passed to the function
parser.add_argument('--input_img',
                    type=str,
                    default=None,
                    help='''input bold data''')

parser.add_argument('--input_img_json',
                    type=str,
                    default=None,
                    help='''json file associated with bold data''')

parser.add_argument('--denoise_strategy',
                    type=str,
                    default=None,
                    help='''denoise strategy label
                    located in denoise_strategies.json''')

parser.add_argument('--filter_strategy',
                    type=str,
                    default=None,
                    help='''filter strategy located in json''')

parser.add_argument('--output_img',
                    type=str,
                    default=None,
                    help='''output file''')


def denoise_img(input_img, input_img_json, confound_strategy,
                filter_strategy, output_img):

    # Interpret the denoise strategy based on the json
    # Load confound strat (assumed to be in same location)
    parameters = json.load(open(os.path.dirname(os.path.realpath(__file__))
                                + '/denoise_config.json',))

    # Get confounds
    confounds, sample_mask = load_confounds_strategy(
        input_img, **parameters[confound_strategy])

    # Load filter strat
    filter_params = parameters[filter_strategy]

    # Get TR
    t_r = json.load(open(input_img_json,))['RepetitionTime']

    # Clean and save out timeseries
    if input_img.endswith('.dtseries.nii'):

        # get timeseries
        img = nb.load(input_img)
        timeseries = img.get_fdata()

        # clean the timeseries
        # note: filtering is already done on the confounds
        clean_timeseries = clean(
            timeseries,
            detrend=True,
            standardize="zscore_sample",
            confounds=confounds,
            high_pass=filter_params['high_pass'],
            low_pass=filter_params['low_pass'],
            t_r=t_r,
            sample_mask=sample_mask,
            kwargs={'clean__filter': filter_params['filter']}
        )

        # save out accounting for lose of timepoints with
        # sample_mask
        new_header = (
            nb.cifti2.cifti2_axes.SeriesAxis(
                start=0.0,
                step=t_r,
                size=clean_timeseries.shape[0],
                unit='second'),
            img.header.get_axis(1)
        )

        nb.save(nb.Cifti2Image(clean_timeseries,
                               header=new_header,
                               nifti_header=img.nifti_header), output_img)

    elif input_img.endswith('.nii.gz'):
        # Clean the timeseries
        cleaned_img = clean_img(input_img,
                                detrend=True,
                                standardize='zscore_sample',
                                confounds=confounds,
                                high_pass=filter_params['high_pass'],
                                low_pass=filter_params['low_pass'],
                                t_r=t_r,
                                kwargs={'clean__sample_mask': sample_mask,
                                        'clean__filter':
                                        filter_params['filter']}
                                )
        # save out
        nb.save(cleaned_img, output_img)
    return output_img


if __name__ == '__main__':
    # Read in user-specified parameters
    args = parser.parse_args()

    denoise_img(args.input_img,
                args.denoise_strategy,
                args.filter_strategy,
                args.output_img)
