from data_download import load_data  # 复用您原本的 data_download
import os


def get_dataloader(args):
    data_path = os.path.join(args.root_path, 'data')
    save_path = os.path.join(args.root_path, 'results')
    if not os.path.exists(save_path): os.makedirs(save_path)

    # 调用原有的加载函数
    tt, tp, num_classes, _, channel, _, dst, input_size, idx_shuffle, mean_std = load_data(
        dataset=args.dataset, root_path=args.root_path, data_path=data_path, save_path=save_path)

    return tt, dst, idx_shuffle, num_classes