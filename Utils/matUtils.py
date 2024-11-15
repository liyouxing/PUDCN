import os
import scipy.io as sio


def load_mat(mat_path):
    """
        .mat要求：文件中保存的是单个结构体
        func: 直接将.mat中的单个结构体转换成numpy
    """
    data = sio.loadmat(mat_path)[mat_path.split('/')[-1][:-4]]  # -4表示去除.mat后缀名, (h, w)矩阵，读取的还是 (h, w)

    return data


def load_mat_key(mat_path, item_name):
    """
        .mat要求：文件中保存的是结构体的结构体
        func: 在结构体中找出name满足item_name的子结构体
    """
    data = sio.loadmat(mat_path)
    data_key = list(data.keys())[-1]
    data_item = data[data_key][0, 0][item_name]
    return data_item


def load_mats(mat_path):
    """
        .mat要求：文件中保存的是结构体的结构体
        func: 输出结构体
    """
    data = sio.loadmat(mat_path)
    data_key = list(data.keys())[-1]
    data_item = data[data_key][0, 0]
    return data_item


def mat_split(mat_dir, save_dir):
    """
        将.mat中的批量文件分开，并保存在save_dir路径下
    """

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # load .mat files
    m_data = load_mats(mat_dir)  # load .mat data

    length = len(m_data)

    for i in range(length):  # 将每个GT分别保存在单独的mat文件中
        name = 'angle' + str(2 * i)
        sio.savemat(save_dir + name + '.mat', {name: m_data[name]}, do_compression=True)
        print(name)


def tensor_to_array(tensor):
    return tensor.cpu().numpy()


def save_ts3_to_mat(tensor, save_dir, save_name, key=None):
    """
        tensor: (1, c, h, w)
        save_dir: save_path _ save_name
    """

    np_array = tensor_to_array(tensor).squeeze(0)
    if tensor.size()[1] == 1:  # c == 1  # 去除所有维度为1的维度
        np_array = np_array.squeeze(0)

    if key is None:
        sio.savemat(save_dir + save_name[:-4] + '.mat', {save_name[:-4]: np_array}, do_compression=True)
    else:
        sio.savemat(save_dir + save_name[:-4] + '.mat', {key: np_array}, do_compression=True)


if __name__ == "__main__":
    pass
