import numpy as np
from scipy.spatial.transform import Rotation as R

order_map = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

reverse_order_map = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation'
}

offset_lists = {
    'Bip001': np.array([0., 0., 0.]),
    'Bip001_Pelvis': np.array([0., -0.041405, 0.046743]),
    "Bip001_Spine": np.array([0., 0.044697, 0.077000]),
    'Bip001_Spine1': np.array([0., 0., 0.153780]),
    'Bip001_Spine2': np.array([0., 0.018365, 0.152706]),
    'Bip001_Neck': np.array([0., -0.043732, 0.211513]),
    'Bip001_Head': np.array([0., -0.024144, 0.056390]),
    'Bip001_L_Clavicle': np.array([0.048991, -0.036560, 0.132514]),
    'Bip001_L_UpperArm': np.array([0.164078, 0.002989, -0.026916]),
    'Bip001_L_Forearm': np.array([0.280778, -0.064501, -0.153731]),
    'Bip001_L_Hand': np.array([0.225800, -0.120182, -0.130038]),
    'Bip001_R_Clavicle': np.array([-0.048991, -0.036560, 0.132514]),
    'Bip001_R_UpperArm': np.array([-0.164078, 0.002989, -0.026915]),
    'Bip001_R_Forearm': np.array([-0.280778, -0.064501, -0.153732]),
    'Bip001_R_Hand': np.array([-0.225800, -0.120182, -0.130038]),
    'Bip001_L_Thigh': np.array([0.082956, 0., 0.]),
    'Bip001_L_Calf': np.array([0.014831, 0.008289, -0.536358]),
    'Bip001_L_Foot': np.array([0.009434, 0.072111, -0.475449]),
    'Bip001_R_Thigh': np.array([-0.082956, 0., 0.]),
    'Bip001_R_Calf': np.array([-0.014831, 0.008289, -0.536358]),
    'Bip001_R_Foot': np.array([-0.009434, 0.072111, -0.475449])
}

def align_quat(qt: np.ndarray, inplace: bool):
    ''' make q_n and q_n+1 in the same semisphere
        the first axis of qt should be the time
    '''
    qt = np.asarray(qt)
    if qt.shape[-1] != 4:
        raise ValueError('qt has to be an array of quaterions')

    if not inplace:
        qt = qt.copy()

    if qt.size == 4:  # do nothing since there is only one quation
        return qt

    sign = np.sum(qt[:-1] * qt[1:], axis=-1)
    sign[sign < 0] = -1
    sign[sign >= 0] = 1
    sign = np.cumprod(sign, axis=0)

    qt[1:][sign < 0] *= -1
    return qt

class JointInfo:
    def __init__(self, name, offset, channels):
        self.name = name
        self.offset = offset
        self.channels = channels
        
class Joint:
    def __init__(self, info: JointInfo, idx):
        self.info = info
        self.idx = idx
        self.children = []

    def is_leaf(self):
        return len(self.children) == 0

class Skeleton:
    def __init__(self, root: Joint):
        self.root = root

    @staticmethod
    def build_skeleton(names, parents, offsets, channels):
        joints = []
        for i in range(len(names)):
            joint_info = JointInfo(names[i], offsets[i], channels[i])
            joint = Joint(joint_info, i)
            joints.append(joint)
        for i in range(len(parents)):
            if parents[i] == -1:
                root = joints[i]
            else:
                joints[parents[i]].children.append(joints[i])
        return Skeleton(root)
    
    def reset_root_by_name(self, root_name):
        new_root = self._reset_root_by_name(self.root, root_name)
        if new_root is None:
            raise ValueError('Root not found')
        self.root = new_root

    def _reset_root_by_name(self, joint: Joint, root_name):
        if joint.info.name == root_name:
            return joint
        
        for child in joint.children:
            new_root = self._reset_root_by_name(child, root_name)
            if new_root is not None:
                return new_root
        return None
    
    def pre_order_traversal(self, callback):
        self._pre_order_traversal(self.root, callback)

    def _pre_order_traversal(self, joint: Joint, callback):
        if joint.is_leaf():
            callback(joint)
            return
        
        callback(joint)
        for child in joint.children:
            self._pre_order_traversal(child, callback)
    
    @staticmethod
    def flatten_skeleton(joint: Joint):
        joints = []
        joints.append(joint)
        for child in joint.children:
            joints.extend(Skeleton.flatten_skeleton(child))
        return joints

class Animation:
    '''
    Animation data class.
    '''
    def __init__(self, root_idx, parents, offsets, channels, positions, rotations, frame_time=0.033333):
        '''
        Initialize the animation data.
        Input:
            root_idx: int
            parents: np.ndarray, shape=(num_joints,), dtype=int
            offsets: np.ndarray, shape=(1, num_joints, 3)
            positions: np.ndarray, shape=(num_frames, num_joints, 3)
            rotations: np.ndarray, shape=(num_frames, num_joints, 4)
            frame_time: float, optional, default=0.033333
        '''
        self.root_idx = root_idx
        self.parents = parents
        self.offsets = offsets
        self.channels = channels
        self.positions = positions
        self.transformations = positions
        self.rotations = rotations
        self.frame_time = frame_time
        self.num_frames = positions.shape[0]
        self.reset()

        for i in range(len(self.parents)):
            self.rotations[:, i] = align_quat(self.rotations[:, i], True)
            self.orientations[:, i] = align_quat(self.orientations[:, i], True)

    def reset(self):
        self.transformations, self.orientations = self.fk()

    def fk(self, positions=None, rotations=None):
        '''
        Forward kinematics.
        '''
        if positions is None:
            positions = self.positions.copy()
        if rotations is None:
            rotations = self.rotations.copy()

        transformations = np.zeros_like(positions)
        orientations = np.zeros_like(rotations)
        for i in range(len(self.parents)):
            if self.parents[i] == -1:
                transformations[:, i] = positions[:, i]
                orientations[:, i] = rotations[:, i]
            else:
                offset = self.offsets[0, i]
                parent_idx = self.parents[i]
                transformations[:, i] = transformations[:, parent_idx] + R.from_quat(orientations[:, parent_idx]).apply(offset)
                orientations[:, i] = (R.from_quat(orientations[:, parent_idx]) * R.from_quat(rotations[:, i])).as_quat()

        return transformations, orientations

    def __len__(self):
        return self.num_frames
    
    @property
    def num_joints(self):
        return len(self.parents)
    
    @property
    def data(self):
        return np.concatenate([self.transformations, self.rotations], axis=2)
    
    def sub(self, start, end):
        return Animation(self.root_idx, self.parents, self.offsets, self.channels, self.transformations[start:end], self.rotations[start:end], self.frame_time)
    
    def _reparent(self, joint_list: list[int] | np.ndarray, root_idx=0):
        new_parents = np.zeros(len(joint_list), dtype=int)
        new_parents[root_idx] = -1
        for i in range(len(joint_list)):
            if i == root_idx:
                continue
            new_parents[i] = joint_list.index(self.parents[joint_list[i]])
        return new_parents
    
    def child(self, joint_list: list[int] | np.ndarray, root_idx=0):
        new_parents = self._reparent(joint_list, root_idx)
        new_channels = self.channels[joint_list].copy()
        new_channels[root_idx] = 6
        new_rotations = self.rotations[:, joint_list].copy()
        new_rotations[:, root_idx] = self.orientations[:, joint_list[root_idx]]
        return Animation(root_idx, new_parents, self.offsets[:1, joint_list].copy(), new_channels, self.positions[:, joint_list].copy(), new_rotations, self.frame_time)
    
    def rotate_root(self, rotation: R):
        self.offsets[0, 0] = rotation.apply(self.offsets[0, 0])
        self.positions[:, 0] = rotation.apply(self.positions[:, 0])
        self.rotations[:, 0] = (R.from_quat(self.rotations[:, 0]) * rotation).as_quat()
        self.reset()

    def rotate_offset(self, rotation: R):
        self.offsets[0] = rotation.apply(self.offsets[0])
        self.positions[:, 0] = rotation.apply(self.positions[:, 0])
        # self.rotations[:, 0] = (rotation * R.from_quat(self.rotations[:, 0]) * rotation.inv()).as_quat()
        # for i in range(0, self.num_joints):
        #     self.rotations[:, i] = (rotation * R.from_quat(self.rotations[:, i]) * rotation.inv()).as_quat()
        self.reset()

class BVHParser:
    @staticmethod
    def check_rotation_order(channels):
        '''
        Check the rotation order from the channels.
        Input:
            channels: List[str]

        Output:
            order: str
        '''
        order = ''
        for channel in channels:
            if 'rotation' in str.lower(channel):
                order += order_map[channel]
        return order

    @staticmethod
    def read_bvh(file_path, order='xyz'):
        '''
        Read a BVH file and return the root joint and motion data.
        Input:
            file_path: str
            order: str, optional, default='xyz'

        Output:
            names: List[str]
            animation_data: AnimationData
        '''
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if len(line) > 0]

        # Read the hierarchy
        ptr = 0
        stack = []
        assert 'HIERARCHY' in lines[ptr], 'HIERARCHY not found'
        while 'MOTION' not in lines[ptr]:
            # Read the hierarchy
            if lines[ptr].split()[0] == 'HIERARCHY':
                names = []
                parents = []
                offsets = []
                channels = []
                ptr += 1
                continue

            # Read the root, three lines
            if lines[ptr].split()[0] == 'ROOT':
                root_idx = len(names)
                names.append('_'.join(lines[ptr].split()[1:])) # Root name
                parents.append(-1) # Root parent
                ptr += 1    # Next line
                if '{' in lines[ptr]:
                    stack.append(root_idx)
                    ptr += 1    # Skip '{'
                else:
                    continue
                offsets.append(np.array([float(word) for word in lines[ptr].split()[1:]]))  # Root offset
                ptr += 1    # Skip 'OFFSET'
                num_channels = int(lines[ptr].split()[1])
                assert num_channels in [0, 3, 6], 'Unknown number of channels %d' % num_channels
                assert len(lines[ptr].split()) == 2 + num_channels, 'Number of channels does not match'
                channels.append([word for word in lines[ptr].split()[2:]])  # Root channels
                new_order = BVHParser.check_rotation_order(channels[-1])
                if new_order != '' and order != new_order:
                    raise ValueError('Rotation order does not match')
                ptr += 1    # Skip 'CHANNELS'
            elif lines[ptr].split()[0] == 'JOINT':
                idx = len(names)
                names.append('_'.join(lines[ptr].split()[1:])) # Joint name
                parents.append(stack[-1])   # Parent is the last element in the stack
                ptr += 1    # Next line
                if '{' in lines[ptr]:
                    stack.append(idx)
                    ptr += 1    # Skip '{'
                else:
                    continue
                offsets.append(np.array([float(word) for word in lines[ptr].split()[1:]]))
                ptr += 1    # Skip 'OFFSET'
                num_channels = int(lines[ptr].split()[1])
                assert num_channels in [0, 3, 6], 'Unknown number of channels %d' % num_channels
                assert len(lines[ptr].split()) == 2 + num_channels, 'Number of channels does not match'
                channels.append([word for word in lines[ptr].split()[2:]])  # Joint channels
                new_order = BVHParser.check_rotation_order(channels[-1])
                if new_order != '' and order != new_order:
                    raise ValueError('Rotation order does not match')
                ptr += 1    # Skip 'CHANNELS'
            elif lines[ptr].split()[0] == 'End':
                names.append('End Site')
                parents.append(stack[-1])
                ptr += 1    # Skip 'End Site'
                ptr += 1    # Skip '{'
                offsets.append(np.array([float(word) for word in lines[ptr].split()[1:]]))
                channels.append([])
                ptr += 1    # Skip 'OFFSET'
                ptr += 1    # Skip '}'
            elif '}' in lines[ptr]:
                stack.pop()
                ptr += 1    # Skip '}'
            else:
                raise ValueError('Unknown line: %s' % lines[ptr])
        offsets = np.array(offsets).reshape(1, len(offsets), 3)
        parents = np.array(parents)
            
        # Read the motion data
        assert 'MOTION' in lines[ptr]
        positions = []
        rotations = []
        channel_nums = np.array([len(channel) for channel in channels])
        ptr += 1
        while ptr < len(lines):
            if 'Frames:' in lines[ptr]:
                num_frames = int(lines[ptr].split()[1])
            elif 'Frame Time:' in lines[ptr]:
                frame_time = float(lines[ptr].split()[2])
            else:
                record = [float(word) for word in lines[ptr].split()]
                current_positions = []
                current_rotations = []
                for channel in channels:
                    joint_rotation = np.zeros(3)
                    joint_position = np.zeros(3)
                    for i, channel_name in enumerate(channel):
                        if 'position' in str.lower(channel_name):
                            joint_position[i % 3] = record.pop(0)
                        elif 'rotation' in str.lower(channel_name):
                            joint_rotation[i % 3] = record.pop(0)
                    current_positions.append(joint_position)
                    current_rotations.append(joint_rotation)
                assert len(record) == 0, 'Record not empty'
                positions.append(np.array(current_positions))
                rotations.append(np.array(current_rotations))
            ptr += 1

        for i in range(len(rotations)):
            rotations[i] = R.from_euler(order, rotations[i], degrees=True).as_quat()
        positions = np.array(positions)
        rotations = np.array(rotations)

        return names, Animation(root_idx, parents, offsets, channel_nums, positions, rotations, frame_time)


    @staticmethod
    def write_bvh(file_path, names, motion_data: Animation, order='xyz'):
        '''
        Write a BVH file.
        Input:
            file_path: str
            names: List[str]
            channels: List[int]
            motion_data: AnimationData
            order: str, optional, default='xyz'
        '''
        with open(file_path, 'w') as f:
            f.write('HIERARCHY\n')
            skeleton = Skeleton.build_skeleton(names, motion_data.parents, motion_data.offsets[0], motion_data.channels)
            BVHParser.write_joint(f, skeleton.root, 0, 'ROOT', order)
            f.write('MOTION\n')
            f.write('Frames: %d\n' % motion_data.num_frames)
            f.write('Frame Time: %f\n' % motion_data.frame_time)
            for frame in motion_data.data:
                frame_data = []
                for jid, joint in enumerate(frame):
                    if motion_data.channels[jid] == 6:
                        frame_data.extend(joint[:3])
                        frame_data.extend(R.from_quat(joint[3:]).as_euler(order, degrees=True))
                    elif motion_data.channels[jid] == 3:
                        frame_data.extend(R.from_quat(joint[3:]).as_euler(order, degrees=True))
                f.write(' '.join([str(value) for value in frame_data]) + '\n')

    @staticmethod
    def write_joint(f, joint: Joint, depth=0, type='ROOT', order='xyz'):
        '''
        Write a joint to a BVH file.
        Input:
            f: file
            joint: Joint
            depth: int, optional, default=0
            type: str, optional, default='ROOT'
            order: str, optional, default='xyz'
        '''
        f.write('\t' * depth + f'{type} %s\n' % joint.info.name)
        f.write('\t' * depth + '{\n')
        f.write('\t' * (depth + 1) + 'OFFSET %f %f %f\n' % (joint.info.offset[0], joint.info.offset[1], joint.info.offset[2]))

        if (joint.info.channels == 6):
            channels = ' '.join(['Xposition', 'Yposition', 'Zposition'] + [reverse_order_map[order[i]] for i in range(3)])
        elif (joint.info.channels == 3):
            channels = ' '.join([reverse_order_map[order[i]] for i in range(3)])
        elif (joint.info.channels == 0):
            channels = ''
        else:
            raise ValueError('Unknown number of channels %d' % joint.info.channels)
        
        if joint.info.channels > 0:
            f.write('\t' * (depth + 1) + 'CHANNELS %d %s\n' % (joint.info.channels, channels))
        else:
            f.write('\t' * (depth + 1) + 'CHANNELS 0\n')
        
        if joint.is_leaf() and joint.info.name != 'End Site':
            f.write('\t' * (depth + 1) + 'End Site\n')
            f.write('\t' * (depth + 1) + '{\n')
            f.write('\t' * (depth + 2) + 'OFFSET %f %f %f\n' % (0, 0, 0))
            f.write('\t' * (depth + 1) + '}\n')
        for child in joint.children:
            if child.info.name == 'End Site':
                f.write('\t' * (depth + 1) + 'End Site\n')
                f.write('\t' * (depth + 1) + '{\n')
                f.write('\t' * (depth + 2) + 'OFFSET %f %f %f\n' % (child.info.offset[0], child.info.offset[1], child.info.offset[2]))
                f.write('\t' * (depth + 1) + '}\n')
            else:
                BVHParser.write_joint(f, child, depth + 1, 'JOINT')
        f.write('\t' * depth + '}\n')


if __name__ == '__main__':
    names, animation_data = BVHParser.read_bvh('test.bvh')
    BVHParser.write_bvh('output.bvh', names, animation_data)