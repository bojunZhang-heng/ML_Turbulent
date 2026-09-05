import numpy as np
import os, json, time
import multiprocessing as mp
import argparse
import h5py
import traceback

from givernylocal.turbulence_dataset import *
from givernylocal.turbulence_toolkit import *

from tqdm import tqdm

from download_setups_3d import get_setup
from render import render_trajectory



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 3D JHTDB Coarse Isotropic Turbulence [1024]")
    parser.add_argument("--out_path", type=str, default="./data/")
    parser.add_argument("--token", type=str, default="NO_TOKEN_PROVIDED")
    parser.add_argument("--temporal_start", type=int, default=1)
    parser.add_argument("--temporal_end", type=int, default=1)

    args = parser.parse_args()

def prepare_data_and_log(json_path_log, h5py_path, p):
    # create log file for this set of simulations
    with open(json_path_log, "w") as f:
        newDict = {"all": p}
        json.dump(newDict, f, indent=4)
        f.close()

    # create h5py file for dataset or overwrite existing parameter
    with h5py.File(h5py_path, "a") as h5py_file:

        print("Opened h5py file at %s" % h5py_path)

        if not "sims" in h5py_file:
            dataset = h5py_file.create_group("sims", track_order=True)
        else:
            dataset = h5py_file["sims"]

        for key in p:
            dataset.attrs[key] = p[key]

        if not "sims/sim0" in h5py_file:
            time_steps = 1 + ((p["Temporal End"] - p["Temporal Start"]) // p["Temporal Step"])
            time_start = p["Temporal Start"] // p["Temporal Step"]
            channels = len(p["Fields"])
            res_x = 1 + ((p["Spatial End"][0] - p["Spatial Start"][0]) // p["Spatial Step"][0])
            res_y = 1 + ((p["Spatial End"][1] - p["Spatial Start"][1]) // p["Spatial Step"][1])
            res_z = 1 + ((p["Spatial End"][2] - p["Spatial Start"][2]) // p["Spatial Step"][2])
            print("Created empty dataset with shape (%d, %d, %d, %d, %d)" % (time_steps, res_x, res_y, res_z, channels))
            for i in range(time_steps):
                print(f"sims/sim0/{i+time_start}")
                h5py_file.create_dataset(f"sims/sim0/{i+time_start}", shape=(res_x, res_y, res_z, channels), dtype='float16',
                                                      chunks=(64, 64, 64, channels))

        h5py_file.close()


def core_download(token, data_set, output_path, field_code, axes_ranges, strides):

    dataset = turb_dataset(dataset_title=data_set, output_path=output_path, auth_token=token)

    result = []

    for field in field_code:

        downloaded = getCutout(
            dataset,
            field,
            axes_ranges,
            strides,
            verbose=True
            )

        downloaded_data = downloaded[f'{field}_{str(axes_ranges[-1][0]).zfill(4)}']
        result += [downloaded_data]

    result = np.concatenate(result, axis=-1)

    print("Downloaded slice ", axes_ranges[0,0])
    return axes_ranges[0,0], result



def wrapped_download(*args, **kwargs):
    try:
        return core_download(*args, **kwargs)
    except Exception as e:
        raise e
        print("\n\n\n\n\n%" % (traceback.format_exc()))


def download_data(
        dataset_name: str,
        out_name: str,
        out_path: str,
        token: str,
        temporal_start: int = 1,
        temporal_end: int = 1,

    ):

    output_path = "/tmp/"
    download_tries = 10

    # timeout_min = 40 if dataset_name == "mhd1024" else 20
    timeout_min = 120

    poll_interval_min = 0.5
    workers = 3


    # create directories, paths, and logs
    out_dir = os.path.join(out_path, out_name)
    out_dir = out_dir[:-1] if out_dir[-1] == "/" else out_dir
    os.makedirs(out_dir, exist_ok=True)

    json_path_log = os.path.join(out_path, out_name + ".json")
    h5py_path = out_dir + ".hdf5"
    p, field_code = get_setup(dataset_name, temporal_start=temporal_start, temporal_end=temporal_end, spatial_end=[1024,1024,1024])
    prepare_data_and_log(json_path_log, h5py_path, p)

    start = np.array( p["Spatial Start"] ).astype(np.int32)
    end = np.array( p["Spatial End"] ).astype(np.int32)
    step = np.array( p["Spatial Step"] ).astype(np.int32)

    # main download
    data = []
    for t in tqdm(range(p["Temporal Start"], p["Temporal End"]+1, p["Temporal Step"])):
        # check if timestep already exists
        with h5py.File(h5py_path, "r") as h5py_file:
            time_step = (t-1) // p["Temporal Step"]
            data = h5py_file[f"sims/sim0/{time_step}"]
            if np.any(data): # ensure that the timestep is not empty, since h5py intializes with zeros
                h5py_file.close()
                print("Timestep %d already exists, skipping...\n" % t)
                continue
            h5py_file.close()

        # handler for worker processes
        slices = []
        def handle_result(result):
            slices.append(result)
        def handle_error(error):
            print("\n\nError occurred:")
            print(error)

        for tries in range(download_tries):
            # download 2d slices from 3d volume (in parallel) as the JHTDB API does not permit large 3d cutouts
            pool = mp.Pool(workers, maxtasksperchild=1)

            pool_job = []

            for x in range(start[0], end[0]+1, step[0]):
                if x in [s[0] for s in slices]:
                    continue

                slice_start = np.array([x, start[1], start[2]]).astype(np.int32)
                slice_end = np.array([x, end[1], end[2]]).astype(np.int32)
                slice_step = np.array([1, step[1], step[2]]).astype(np.int32)

                x_range = [x, x]
                y_range = [start[1], end[1]]
                z_range = [start[2], end[2]]
                t_range = [t, t]

                strides = np.array([1, step[1], step[2], 1])
                axes_ranges = np.array([x_range, y_range, z_range, t_range])

                pool_job.append(pool.apply_async(wrapped_download, args=(token, dataset_name, output_path, field_code, axes_ranges, strides), callback=handle_result, error_callback=handle_error))

            waited_min = 0
            time.sleep(poll_interval_min*60)
            waited_min += poll_interval_min

            job_finished = [r.ready() for r in pool_job]

            # check regularly if all slices have been downloaded
            while not all(job_finished) and waited_min < timeout_min:
                print("Waiting for slices to download... (%d/%d)" % (len(slices), len(range(start[0], end[0]+1, step[0]))))
                time.sleep(poll_interval_min*60)
                job_finished = [r.ready() for r in pool_job]
                waited_min += poll_interval_min

            # finish if all slices have been downloaded, otherwise retry with new pool
            pool.close()
            if len(slices) >= len(range(start[0], end[0]+1, step[0])):
                pool.join()
                break
            else:
                print("Missing slices, retrying for the %dth time..." % tries)
                pool.terminate()
                pool.join()
                if tries == download_tries-1:
                    raise TimeoutError("Failed to download all slices after %d tries, aborting..." % download_tries)

        channels = len(p["Fields"])
        res_x = 1 + ((p["Spatial End"][0] - p["Spatial Start"][0]) // p["Spatial Step"][0])
        res_y = 1 + ((p["Spatial End"][1] - p["Spatial Start"][1]) // p["Spatial Step"][1])
        res_z = 1 + ((p["Spatial End"][2] - p["Spatial Start"][2]) // p["Spatial Step"][2])

        # write timestep to h5py file
        with h5py.File(h5py_path, "a") as h5py_file:

            slices = sorted(slices, key=lambda x: x[0])
            data = np.concatenate([s[1] for s in slices], axis=2)
            data = np.transpose(data, (2,1,0,3)) # move channels to the front, jhtdb transposes x and z axes

            # data = np.zeros(shape=(channels, res_x, res_y, res_z))

            time_step = (t-1) // p["Temporal Step"]

            h5py_file[f"sims/sim0/{time_step}"][:] = data # np.expand_dims(data, axis=0)

            h5py_file.close()
            print("Timestep %d (index: %d) with shape %s written to disk\n" % (t, (t-1)//p["Temporal Step"], str(data.shape)))


    # reload data partially to render, as full dataset may be too large for RAM

    print(p)

    time_steps = max((p["Temporal End"]+1 - p["Temporal Start"]) // p["Temporal Step"], 1)
    steps_plot = min(10, time_steps)

    data = []
    time_start = p["Temporal Start"] // p["Temporal Step"]
    with h5py.File(h5py_path, "r") as h5py_file:
        for t in range(0, time_steps, time_steps // steps_plot):
            data.append(np.transpose(h5py_file[f"sims/sim0/{t+time_start}"], (3, 0, 1, 2)))  # move channels to front
        h5py_file.close()

    if dataset_name == "mhd1024":
        vmin, vmax = -0.7, 0.7
    elif dataset_name == "isotropic1024coarse":
        vmin, vmax = -1.2, 1.2
    else:
        # automatically determined via min and max of data
        vmin, vmax = None, None

    print(vmin, vmax)

    render_trajectory(
        data=data,
        dimension=3,
        output_path=out_dir,
        sim_id=0,
        time_steps=len(data),
        steps_plot=len(data),
        vmin=vmin,
        vmax=vmax,
    )




if __name__ == '__main__':
    download_data("isotropic1024coarse", "isotropic1024coarse", args.out_path, args.token,
                  args.temporal_start, args.temporal_end)