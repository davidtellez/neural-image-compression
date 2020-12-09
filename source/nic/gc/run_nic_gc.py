import argparse

from nic.gc.run_gc_alg import *
import SimpleITK as sitk
import numpy as np
import shutil

title = "Neural Image Compression"

if __name__ == '__main__':

    input_dir = "/home/witali/projects2/nicfeaturize/test"
    out_dir = "/home/witali/projects/gc-algo/out"
    title = "Neural Image Compression"
    # title = "neural-image-compression"

    os.environ["GCTOKEN"] = "46dfeb370c59a84ece4219d7af9c7911710f3131"

    # files = [f.resolve() for f in Path(input()).iterdir()]
    # print('processing %d files' % len(files))
    #
    # c = Client(token="46dfeb370c59a84ece4219d7af9c7911710f3131")
    # session = c.upload_cases(files=files, algorithm="neural-image-compression")
    #
    # files = [f.resolve() for f in Path("/path/to/files").iterdir()]

def convert_mha_to_npy(path, delete=True, overwrite=False):
    """ converts the mha in path to npy, deletes mha if delete=True. path can be a dir with mhas or an mha """
    print('converting mha to npy...')
    path = Path(path)
    if path.is_dir():
        in_pathes = [p for p in path.iterdir() if p.is_file() and p.suffix in '.mha']
        out_dir = path
    else:
        in_pathes = []
        out_dir = path.parent
        if path.suffix == '.mha':
            in_pathes.append(path)
    if len(in_pathes)==0:
        print('no .mha in %s' % path)

    print('converting %d .mha in %s' % (len(in_pathes), path))
    for p in in_pathes:
        out_path = out_dir/(p.stem+'.npy')
        if out_path.exists() and not overwrite:
            print('skipping existing %s' % out_path)
        else:
            _convert_mha_to_npy(p, out_path)
        if delete:
            print('deleting %s' % p)
            p.unlink()


def _convert_mha_to_npy(path, out_path, overwrite=False):
    image = sitk.ReadImage(str(path), imageIO="MetaImageIO")
    arr = sitk.GetArrayFromImage(image)
    np.save(out_path, arr)
    print('saved %s with shape %s' % (out_path, str(arr.shape)))

def run_nic_on_gc(input_path, output_dir, token=None, upload_session_wait=1200, job_wait=8400):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    allowed_ext = ['.tif', '.tiff']
    input_path = Path(input_path)
    if input_path.is_dir():
        in_pathes = [p for p in input_path.iterdir() if p.is_file() and p.suffix in allowed_ext]
    else:
        in_pathes = []
        if input_path.suffix in allowed_ext:
            in_pathes.append(input_path)

    if len(in_pathes)==0: raise ValueError('no pathes found in %s' % str(input_path))
    print('processing %d pathes' % len(in_pathes))
    for i,p in enumerate(in_pathes):
        print('%d/%d: %s' % (i+1, len(in_pathes), str(p)))
        overlay_out_pathes = cli(input_path=p, output_dir=output_dir, algorithm_title=title,
                                              upload_session_wait=upload_session_wait, job_wait=job_wait,
                                              token=token)
        slide_name = p.stem
        #taken care of in JobExecutor
        # gc removes the input name, so here add it again
        # for i in range(len(overlay_out_pathes)):
        #     op = overlay_out_pathes[i]
        #     if slide_name not in str(op):
        #         renamed = output_dir / (slide_name + '_' + op.name)
        #         shutil.move(op, renamed)
        #         print('renamed %s to %s' % (op.name, renamed.name))
        #         overlay_out_pathes[i] = renamed

    convert_mha_to_npy(output_dir, overwrite=False)
    print('Done!')

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-i', '--input_path', required=True, type=str, help='input dir or image')
    argument_parser.add_argument('-o', '--output_dir', required=True, type=str, help='output directory')
    argument_parser.add_argument('-t', '--token', required=False, type=str, help='grandchallenge token, alternatively set env var GCTOKEN')
    arguments = vars(argument_parser.parse_args())
    run_nic_on_gc(**arguments)
