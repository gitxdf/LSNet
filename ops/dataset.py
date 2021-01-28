# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import torch
import cv2
import av

from . import transforms_video as transform
from torchvision import transforms

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


#dfxue 2020.12.01 store the data_type='video' video info, include path and label
class VideoInfo(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB', data_type='image', interval=0, num_frame=3,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        self.data_type = data_type
        self.interval = interval
        self.num_frame = num_frame
        self.video_num = 0
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(self.root_path, '{:06d}'.format(int(directory)), self.image_tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                    flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        if (not self.test_mode or self.remove_missing) and self.data_type == 'image':
            tmp = [item for item in tmp if int(item[1]) >= 3]
        #dfxue 2020.12.01
        if self.data_type == 'image':
            self.video_list = [VideoRecord(item) for item in tmp]
        elif self.data_type == 'video':
            self.video_list = [VideoInfo(item) for item in tmp]
        else:
            print("data_type {} is not available.\n".format(self.data_type))

        if self.data_type == 'image' and self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            for v in self.video_list:
                v._data[1] = int(v._data[1]) / 2
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    # dfxue 2020.12.01 decode the video in record
    def _decode(self, record, backend='pyav'):
        frames = {}
        filename = '/backup/dfxue/kinetics/kinetics400_videos/error_video.txt'
        if not os.path.exists(record.path):
            with open(filename, 'a') as file_object:
                print("the video is not exists: {}\n".format(record.path))
                file_object.write("video not exist: {}\n".format(record.path))
                return None
        if backend == 'pyav':
            container = av.open(record.path, 'r')
            if len(container.streams.video) == 0:
                print("video error, the path is: {}".format(record.path))
                with open(filename, 'a') as file_object:
                    file_object.write("video error: {}\n".format(record.path))
                return None
            for frame in container.decode(container.streams.video[0]):
                frames[frame.pts] = frame
                # for test
                # frame.to_image().save(
                #     'images/image{}_{:04d}.jpg'.format(record.label, frame.pts),
                #     quality=80,
                # )
            result = [frames[pts] for pts in sorted(frames)]
            # print("=={}th video path: {} success.".format(self.video_num, record.path))
            return result

    # dfxue 2020.12.01 get video sample index
    def _get_video_indices(self, frames, sample_method, mode):
        frames_num = len(frames)
        offsets = []
        if sample_method == 'dense':  # i3d dense sample
            sample_pos = max(1, 1 + frames_num - 64)
            t_stride = 64 // self.num_segments
            if mode == 'test':
                start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
                for start_idx in start_list.tolist():
                    offsets += [(idx * t_stride + start_idx) % frames_num for idx in range(self.num_segments)]
            else:
                start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
                offsets = [(idx * t_stride + start_idx) % frames_num for idx in range(self.num_segments)]
            return np.array(offsets)
        elif sample_method == 'tsn':  # tsn sample
            if mode == 'train':
                average_duration = (frames_num - self.new_length + 1) // self.num_segments
                if average_duration > 0:
                    offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                      size=self.num_segments)
                elif frames_num > self.num_segments:
                    offsets = np.sort(randint(frames_num - self.new_length + 1, size=self.num_segments))
                else:
                    offsets = np.zeros((self.num_segments,), dtype=int)
                return offsets + 1
            elif mode == 'val':
                if frames_num > self.num_segments + self.new_length - 1:
                    tick = (frames_num - self.new_length + 1) / float(self.num_segments)
                    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
                else:
                    offsets = np.zeros((self.num_segments,), dtype=int)
                return offsets + 1
            elif mode == 'test':
                tick = (frames_num - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
                return offsets + 1
            else:
                print("the mode {} is not availbole.\n".format(mode))
        elif sample_method == '':

        else:
            print("the sample method {} is not available.\n".format(sample_method))


    def spatial_sampling(
            self,
            frames,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=224,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
            inverse_uniform_sampling (bool): if True, sample uniformly in
                [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
                scale. If False, take a uniform sample from [min_scale,
                max_scale].
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = transform.random_crop(frames, crop_size)
            if random_horizontal_flip:
                frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames

    def __getitem__(self, index):
        record = self.video_list[index]
        index_cross = True
        interval = self.interval
        num_frame = self.num_frame
        length = num_frame * interval - interval + num_frame
        # self.video_num = self.video_num + 1
        #dfxue 2020.12.01 sample frame from video
        if self.data_type == 'video':
            frames = {}
            for _ in range(10):
                # print(">>{}th video path: {}.".format(self.video_num, record.path))
                frames = self._decode(record, backend='pyav')
                if frames is None:
                    print("====>ERROR: {}th video is None, the video is {}.\n".format(self.video_num, record.path))
                    record = self.video_list[np.random.randint(len(self.video_list))]  # index+1
                    # self.video_num = self.video_num + 1
                    continue
                frames = [frame.to_rgb().to_ndarray() for frame in frames]
                # frames = torch.as_tensor(np.stack(frames))
                sample_method = 'dense' if self.dense_sample else 'tsn'
                mode = 'test'
                if not self.test_mode:
                    mode = 'train' if self.random_shift else 'val'

                segment_indices = []
                if index_cross:
                    indices = self._get_video_indices(frames, sample_method=sample_method, mode=mode)
                    for i in indices:
                        if i < length // 2:
                            i = i + length // 2
                        elif i > len(frames) - length // 2 - 1:
                            i = i - length // 2 - 1
                        for j in range(num_frame):
                            segment_indices.append(i - length // 2 + j * (1 + interval))
                else:
                    indices = self._get_video_indices(len(frames)//length, sample_method=sample_method, mode=mode)  # ndarray type
                    for i in indices:
                        for j in range(num_frame):
                            segment_indices.append(i * length - j * (interval + 1) - 1)

                images = [frames[i] for i in segment_indices]
                images = [Image.fromarray(image) for image in images]
                images = self.transform(images)
                return images, record.label-1

        # check this is a legit video folder
        if self.image_tmpl == 'flow_{}_{:05d}.jpg':
            file_name = self.image_tmpl.format('x', 1)
            full_path = os.path.join(self.root_path, record.path, file_name)
        elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
            file_name = self.image_tmpl.format(int(record.path), 'x', 1)
            full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
        else:
            file_name = self.image_tmpl.format(1)
            full_path = os.path.join(self.root_path, record.path, file_name)

        while not os.path.exists(full_path):
            print('################## Not Found:', os.path.join(self.root_path, record.path, file_name))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl.format('x', 1)
                full_path = os.path.join(self.root_path, record.path, file_name)
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl.format(int(record.path), 'x', 1)
                full_path = os.path.join(self.root_path, '{:06d}'.format(int(record.path)), file_name)
            else:
                file_name = self.image_tmpl.format(1)
                full_path = os.path.join(self.root_path, record.path, file_name)

        mode = 'test'
        if not self.test_mode:
            mode = 'train' if self.random_shift else 'val'
            # segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        # else:
            # segment_indices = self._get_test_indices(record)
        sample_method = 'dense' if self.dense_sample else 'tsn'

        segment_indices = []
        if index_cross:
            indices = self._get_video_indices(record.num_frames, sample_method=sample_method, mode=mode)
            for i in indices:
                if i < length // 2 + 1:
                    i = i + length // 2
                elif i > record.num_frames - length // 2:
                    i = i - length // 2
                for j in range(num_frame):
                    segment_indices.append(i - length // 2 + j * (1 + interval))
        else:
            indices = self._get_video_indices(record.num_frames//length, sample_method=sample_method, mode=mode)
            for i in indices:
                for j in range(num_frame):
                    segment_indices.append(i * length - j * (interval + 1))
            segment_indices.sort()

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            # for i in range(self.new_length):
            seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
                # if p < record.num_frames:
                #     p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
