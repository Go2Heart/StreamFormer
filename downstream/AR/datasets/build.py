import os
from torchvision import transforms
from .transforms import *
from .ssv2_siglip import SSVideoSiglipClsDataset, SSRawFrameSiglipClsDataset
from .kinetics_siglip import VideoSiglipClsDataset_sparse

def build_dataset(is_train, test_mode, args):
    print(f'Use Dataset: {args.data_set}')
    if args.data_set in [
            'Kinetics_SIGLIP',
        ]:
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
            prefix = os.path.join(args.prefix, 'videos_train')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
            prefix = os.path.join(args.prefix, 'videos_val')
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
            prefix = os.path.join(args.prefix, 'videos_val')
        func = VideoSiglipClsDataset_sparse

        dataset = func(
            anno_path=anno_path,
            prefix=prefix,
            split=args.split,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        
        nb_classes = args.nb_classes

    elif args.data_set in ['SSV2_SIGLIP']:
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        if args.use_decord:
            func = SSVideoSiglipClsDataset
        else:
            func = SSRawFrameSiglipClsDataset

        dataset = func(
            anno_path=anno_path,
            prefix=args.prefix,
            split=args.split,
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    else:
        print(f'Wrong: {args.data_set}')
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
