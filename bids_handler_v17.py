from bids import BIDSLayout
from nilearn import datasets
from nilearn import input_data
from nilearn.interfaces.fmriprep import load_confounds
import os
import pandas as pd
import numpy as np
import multiprocessing

"""

v16 updates

- proper motion extraction and proper compcor parameters as aligned with McCulloch et al. (2023)
- process all "runs" and indicate this in the title of the outputted csv file


"""

def process_data_bids(bids_root, strategy, atlas_name, save_path, save_data=False, limit_subjects=False):
    print("BIDS root:", bids_root)
    print("Directories and files at BIDS root:", os.listdir(bids_root))

    atlases = {
        'schaefer400': datasets.fetch_atlas_schaefer_2018(n_rois=400),
        'schaefer1000': datasets.fetch_atlas_schaefer_2018(n_rois=1000),
        'schaefer100': datasets.fetch_atlas_schaefer_2018(n_rois=100),
        'yeo17': datasets.fetch_atlas_yeo_2011()
    }
    if atlas_name not in atlases:
        raise ValueError(f"Atlas not supported. Available options: {', '.join(atlases.keys())}")

    atlas_data = atlases[atlas_name]
    atlas_filename = atlas_data['thin_17'] if atlas_name == 'yeo17' else atlas_data.maps

    layout = BIDSLayout(bids_root, validate=False, derivatives=True, absolute_paths=True)
    masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename, standardize=True, verbose=2, smoothing_fwhm=6.0)

    subjects = layout.get_subjects()

    if limit_subjects:
        subjects = subjects[15:34]

    all_dfs = []

    for subject_id in subjects:
        sessions = layout.get_sessions(subject=subject_id) or [None]
        for session in sessions:
            func_files = layout.get(subject=subject_id, session=session, suffix='bold', extension='nii.gz', return_type='filename')
            print(f"Processing subject {subject_id}, session: {session}, found {len(func_files)} functional files")
            if not func_files:
                print(f"No functional files found for subject {subject_id}, session {session}.")
                continue

            for func_file in func_files:
                if "derivatives/fmriprep" not in func_file:
                    print(f"Skipping raw file {func_file}, not a preprocessed file.")
                    continue

                

                # task = layout.get_metadata(func_file).get('TaskName', 'unknown')
                # task = task.replace('/', '').replace(' ', '').strip()


                # Try to get TaskName from metadata
                task = layout.get_metadata(func_file).get('TaskName', None)

                # If missing, extract from filename using regex
                if not task or task.lower() == 'unknown':
                    import re
                    task_match = re.search(r'task-([a-zA-Z0-9_]+)', os.path.basename(func_file))
                    if task_match:
                        task = task_match.group(1)
                    else:
                        task = 'unknown'

                task = task.replace('/', '').replace(' ', '').strip()

                print(f"TaskName for file {func_file}: {task}")

                # extract run information
                run ="1"

                metadata_entities = layout.parse_file_entities(func_file) # try bids splitting dictionary in components first
                if 'run' in metadata_entities and metadata_entities['run'] is not None:
                    run = metadata_entities['run']
                else:
                    import re # regular expressions way in case bids fails to find the run
                    run_match = re.search(r'run-(\d+)', func_file)
                    if run_match:
                        run = run_match.group(1) # capture only the digit of run-(\d+)
                print(f"run for file {func_file}: {run}")

                # Try to parse metadata entities 
                metadata_entities = layout.parse_file_entities(func_file)

                # Fallback run parsing
                run = "1"
                if 'run' in metadata_entities and metadata_entities['run'] is not None:
                    run = metadata_entities['run']
                else:
                    import re
                    run_match = re.search(r'run-(\d+)', func_file)
                    if run_match:
                        run = run_match.group(1)
                print(f"run for file {func_file}: {run}")

                # Extract task name from metadata or filename
                task = layout.get_metadata(func_file).get('TaskName', None)
                if not task or task.lower() == 'unknown':
                    import re
                    task_match = re.search(r'task-([a-zA-Z0-9_]+)', os.path.basename(func_file))
                    if task_match:
                        task = task_match.group(1)
                    else:
                        task = 'unknown'
                task = task.replace('/', '').replace(' ', '').strip()
                print(f"TaskName for file {func_file}: {task}")

                # Load confounds with a fallback in case of error
                try:
                    if strategy == 'gsr':
                        confounds, sample_mask = load_confounds(
                            func_file,
                            strategy=["motion", "global_signal", "scrub", "high_pass", "wm_csf"],
                            motion='full', global_signal='basic', scrub=0,
                            fd_threshold=0.5, std_dvars_threshold=1.5
                        )
                    elif strategy == 'compcor':
                        confounds, sample_mask = load_confounds(
                            func_file,
                            strategy=['motion', "high_pass", "scrub", "compcor", "wm_csf"],
                            motion="full", compcor='anat_combined', n_compcor=5,
                            scrub=0, fd_threshold=0.5, std_dvars_threshold=1.5
                        )
                except TypeError as e:
                    print(f"[WARNING] load_confounds failed for {func_file} with error: {e}. Using fallback confounds file.")
                    # Construct the expected confounds TSV filename
                    confounds_file = func_file.replace("_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", "_desc-confounds_timeseries.tsv")
                    if os.path.exists(confounds_file):
                        try:
                            # Load the confounds file manually
                            confounds_df = pd.read_csv(confounds_file, sep='\t')
                            confounds = confounds_df  # use the dataframe as your confounds
                            sample_mask = np.arange(len(confounds_df))
                            print(f"[INFO] Fallback: loaded confounds from {confounds_file}")
                        except Exception as e2:
                            print(f"[ERROR] Fallback failed to load confounds from {confounds_file}: {e2}")
                            confounds = None
                            sample_mask = None
                    else:
                        print(f"[ERROR] Fallback confounds file {confounds_file} does not exist.")
                        confounds = None
                        sample_mask = None

                # Clean confounds: Replace infs and NaNs so that the array is valid
                if confounds is not None:
                    confounds = confounds.replace([np.inf, -np.inf], np.nan)
                    confounds = confounds.fillna(0)

                # Extract time series with cleaned confounds
                time_series = masker.fit_transform(func_file, confounds=confounds, sample_mask=sample_mask)

                # Extract time series
                time_series = masker.fit_transform(func_file, confounds=confounds, sample_mask=sample_mask)
                # to the masker fit transform can be added from the tsv 
                if len(time_series) == 0:
                    print(f"Empty time series for file {func_file}.")
                    continue
                print(f"Time series shape for file {func_file}: {time_series.shape}")

                # Find corresponding confounds file
                confounds_file = func_file.replace("_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz", "_desc-confounds_timeseries.tsv")
                try:
                    confounds_df = pd.read_csv(confounds_file, sep='\t')
                    print(f"Confounds columns for {func_file}: {confounds_df.columns}")
                except Exception as e:
                    print(f"Error reading confounds file {confounds_file}: {str(e)}")
                    continue

                #calculate number of rows of confounds_df
                number_of_rows_confounds = len(confounds_df) 

                
                # Apply sample mask to confounds
                # if sample_mask is not None:
                #     sample_mask = list(sample_mask)  # Convert sample_mask to list to avoid errors with indexing
                #     confounds_df = confounds_df.iloc[sample_mask]

                # # Check if lengths match and trim if necessary
                # if len(confounds_df) != len(time_series):
                #     min_len = min(len(confounds_df), len(time_series))
                #     time_series = time_series[:min_len]
                #     confounds_df = confounds_df.iloc[:min_len]

                # Add framewise displacement
                if 'framewise_displacement' in confounds_df.columns:
                    fd = confounds_df['framewise_displacement']
                else:
                    print(f"Warning: 'framewise_displacement' not found in confounds for {func_file}")

                # calculate mean framewise displacement
                if fd is not None: 
                    fd_mean = np.mean(fd)
                    print(f"Mean framewise displacement for {func_file}: {fd_mean}")

                # create data frame for saving to csv
                df = pd.DataFrame(time_series, columns=[f'Region_{i}' for i in range(time_series.shape[1])])

                # proportion of time series to keep

                # proportion of time series to keep
                if sample_mask is not None:
                    proportion_ts_fd = 1 - (len(sample_mask) / number_of_rows_confounds)
                else:
                    proportion_ts_fd = 0  # No exclusion
                #print(proportion_ts_fd)

                
                df['proportion_ts_fd'] = proportion_ts_fd
                df['FD_mean'] = fd_mean


                # Create DataFrame just for printing
                final_df = df.copy()
                #final_df['FramewiseDisplacement'] = fd.values
                final_df['Dataset'] = os.path.basename(bids_root)
                final_df['Subject'] = subject_id
                final_df['Session'] = session
                final_df['Run'] = run
                final_df['Task'] = task
                final_df['proportion_ts_fd'] = proportion_ts_fd
                final_df['FD_mean'] = fd_mean

                all_dfs.append(final_df)
                #print(final_df.head())

                # Save CSV cotaining time series and framewise displacement with metadata in title
                if save_data:
                    dataset = os.path.basename(bids_root)
                    task = task.replace("space", "").strip("_")
                    csv_filename = f"dat-{dataset}_sub-{subject_id}_task-{task}_ses-{session}_run-{run}_atlas-{atlas_name}_conreg-{strategy}.csv"
                    df.to_csv(os.path.join(save_path, csv_filename), index=False)
                    print(f"Saved data for dataset {dataset} to {csv_filename}")

    final_combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    #print(final_combined_df.head())
    return final_combined_df



def process_root(root, strategy, atlas_name, save_path):
    print(f"Processing {root}")
    process_data_bids(root, strategy, atlas_name, save_path, save_data=True, limit_subjects=True)
    print(f"Finished processing {root}")

def safe_process_root(root, strategy, atlas_name, save_path):
    try:
        process_root(root, strategy, atlas_name, save_path)
    except Exception as e:
        print(f"[ERROR] Failed to process {root}: {e}")

if __name__ == '__main__':
    roots = [
        r"/home/s1836706/data_pi-michielvanelkm/a_unaltered_SCDB/pennstate_sleep"
        # r"/home/s1836706/data_pi-michielvanelkm/Parsa/null_simulation/sim_Data/basel_LAM",
        # r"/home/s1836706/data_pi-michielvanelkm/Parsa/null_simulation/sim_Data/basel_lsd",
        # r"/home/s1836706/data_pi-michielvanelkm/Parsa/null_simulation/sim_Data/basel_MMM",
        # r"/home/s1836706/data_pi-michielvanelkm/Parsa/null_simulation/sim_Data/basel_PLM",
        # r"/home/s1836706/data_pi-michielvanelkm/Parsa/null_simulation/sim_Data/berlin_fls-ganz",
        # r"/home/s1836706/data_pi-michielvanelkm/Parsa/null_simulation/sim_Data/london_lsd",
        # r"/home/s1836706/data_pi-michielvanelkm/Parsa/null_simulation/sim_Data/london_psilo",
        # r"/home/s1836706/data_pi-michielvanelkm/Parsa/null_simulation/sim_Data/maastricht_thc-coc",
        # r"/home/s1836706/data_pi-michielvanelkm/Parsa/null_simulation/sim_Data/zurich_meditation",
    ]

    save_path = r"/home/s1836706/data_pi-michielvanelkm/CopBET/EntProjContinuation/pennSleep/csv_preprocessed"
    strategy = "compcor"
    atlas_name = "schaefer100"

    # Create a pool of worker processes
    num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    # Create a list of arguments for each root
    args = [(root, strategy, atlas_name, save_path) for root in roots]

    # Map the process_root function to the pool of workers
    pool.starmap(safe_process_root, args)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    print("All roots processed.")

    # code to also do different conreg technique at the same time
    strategy = "gsr"

    # Create a pool of worker processes
    num_processes = multiprocessing.cpu_count()  # Use all available CPU cores
    pool = multiprocessing.Pool(processes=num_processes)

    # Create a list of arguments for each root
    args = [(root, strategy, atlas_name, save_path) for root in roots]

    # Map the process_root function to the pool of workers
    pool.starmap(safe_process_root, args)

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()

    print("All roots processed.")
