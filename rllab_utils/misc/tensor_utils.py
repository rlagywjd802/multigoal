import numpy as np
import inspect

def stack_tensor_list(tensor_list):
    if tensor_list is None or len(tensor_list) == 0:
        # print('WARINING: Sampler: Empty tensor list provided')
        return tensor_list
    # In case we have tuple observations - we will have a list of tuples
    # Thus we have to re-pack it to tuple of lists and then stack them to np array
    if isinstance(tensor_list[0], tuple):
        lists = list(zip(*tensor_list))
        arrays = []
        for lst in lists:
            arrays.append(np.stack(lst))
            # print('!stack_tensor_list: array shape ', arrays[-1].shape)
        return tuple(arrays)
    else:
        stacked_array = np.array(tensor_list)
        # print('!!stack_tensor_list: array shape ', np.array(tensor_list).shape, ' Tensor shape = ', tensor_list[0].shape)
        return np.stack(tensor_list)

def truncate_tensor_list(tensor_list, truncated_len):
    # In case we have tuple observations - we will have a list of tuples
    if isinstance(tensor_list[0], tuple):
        tensor_list = list(tensor_list)
        for lst_i in range(len(tensor_list)):
            tensor_list[lst_i] = tensor_list[lst_i][:truncated_len]
        return tuple(tensor_list)
    else:
        return tensor_list[:truncated_len]


def concat_tensor_list(tensor_list):
    if isinstance(tensor_list[0], tuple):
        obs_lists = list(zip(*tensor_list))
        # print('concat_tensor_list: obs = ', [len(obs) for obs in obs_lists])
        return tuple([np.concatenate(obs, axis=0) for obs in obs_lists])
    else:
        return np.concatenate(tensor_list, axis=0)